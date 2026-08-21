"""
Multi-Head Self Attention (MHSA)
=================================
Standard scaled dot-product multi-head self attention with:
  - Rotary Position Embedding (RoPE) on queries and keys
  - Causal (auto-regressive) masking
  - No bias on projection layers
  - Pre-norm placement handled by the enclosing DecoderBlock

Mathematical definition
-----------------------
Given input X ∈ ℝ^{B×T×d}:

  Q = X Wq,  K = X Wk,  V = X Wv       (projections, no bias)

Split into H heads, each of dimension d_h = d / H:

  Qₕ, Kₕ = RoPE(Qₕ), RoPE(Kₕ)         (apply rotary embeddings)

Scaled dot-product attention per head:

  Aₕ = softmax( (Qₕ Kₕᵀ) / √d_h + mask ) Vₕ

where mask[i,j] = 0 if j ≤ i else −∞  (causal constraint).

Concatenate and project:

  output = concat(A₁, ..., A_H) Wo

Complexity
----------
Time : O(T² · d)  — quadratic in sequence length (standard attention)
Space: O(T² · H)  — attention weight matrix per head

Numerical stability
-------------------
- Scaling by 1/√d_h keeps the pre-softmax logits in a well-conditioned
  range, preventing vanishing gradients from very peaked softmax outputs.
- Softmax is computed by PyTorch's numerically stable implementation
  (subtract max before exp).
- RoPE is applied in float32 (see rope.py).
- Causal mask adds −∞ (not a large negative number) so masked positions
  become exactly 0 after softmax — no gradient leakage.

FlashAttention compatibility
-----------------------------
The forward pass is written in a way that is structurally compatible with
a future drop-in replacement by ``torch.nn.functional.scaled_dot_product_attention``
(PyTorch 2.0+) or the ``flash-attn`` library.  To migrate:
  1. Replace the manual QKᵀ/softmax/V block with:
         F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                        dropout_p=0.0, is_causal=True)
  2. Remove the manual mask addition (is_causal=True handles it).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cyberslm.model.config import CyberSLMConfig
from cyberslm.model.rope import RotaryPositionEmbedding, apply_rope


def _causal_bias(
    q_len: int,
    k_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Additive causal mask of shape ``(q_len, k_len)`` for a query block that
    starts at absolute position ``past_len``.

    Query row ``i`` represents absolute position ``past_len + i`` and may attend
    to key columns ``0 .. past_len + i`` inclusive; everything after is -inf.
    With ``past_len == 0`` this reduces to the usual upper-triangular mask.
    """
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + past_len  # (q,1)
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)             # (1,k)
    return torch.where(
        k_pos <= q_pos,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), float("-inf"), dtype=dtype, device=device),
    )


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention with RoPE and causal masking.

    This module owns the four projection matrices (Wq, Wk, Wv, Wo),
    the RoPE cache, and the causal mask buffer.

    Parameters
    ----------
    config : CyberSLMConfig
        Validated model configuration.

    Attributes
    ----------
    q_proj : nn.Linear   ``(hidden_dim, hidden_dim)``, no bias
    k_proj : nn.Linear   ``(hidden_dim, hidden_dim)``, no bias
    v_proj : nn.Linear   ``(hidden_dim, hidden_dim)``, no bias
    o_proj : nn.Linear   ``(hidden_dim, hidden_dim)``, no bias
    rope   : RotaryPositionEmbedding

    Shape
    -----
    Input  : ``(batch, seq_len, hidden_dim)``
    Output : ``(batch, seq_len, hidden_dim)``
    """

    def __init__(
        self,
        config: CyberSLMConfig,
        rope: Optional[RotaryPositionEmbedding] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads  = config.num_heads
        self.head_dim   = config.head_dim
        self.scale      = 1.0 / math.sqrt(self.head_dim)
        self.attn_dropout_p = config.attn_dropout

        # ------------------------------------------------------------------ #
        # Projection layers — no bias (modern practice, saves ~4×384 params) #
        # ------------------------------------------------------------------ #
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # ------------------------------------------------------------------ #
        # RoPE cache                                                          #
        # ------------------------------------------------------------------ #
        # The cos/sin tables depend only on (head_dim, max_seq_len, base), so
        # every layer's would be byte-identical. CyberSLM builds ONE and passes
        # it in; previously each of the 12 layers constructed its own, costing
        # ~12 MB of duplicated buffers. Falls back to building its own so the
        # module stays usable standalone (tests, ablations).
        self.rope = rope if rope is not None else RotaryPositionEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        # Causality is enforced by scaled_dot_product_attention(is_causal=...)
        # rather than a materialised (max_seq_len × max_seq_len) mask buffer,
        # which previously cost ~67 MB per layer.

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """
        Compute multi-head self attention.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, seq_len, hidden_dim)``.
        attention_mask : Optional[Tensor]
            Key-padding mask of shape ``(batch, seq_len)`` with 1 for real
            tokens and 0 for padding. When provided, padded keys are excluded
            from every query's attention (in addition to the causal mask).
            ``None`` means no padding (the common training case with packed
            sequences).
        return_attn_weights : bool
            If True, also return the attention weight matrix for inspection.
            This forces the slower explicit-softmax path; leave False for
            training so the fused kernel is used.

        Returns
        -------
        output : Tensor
            Shape ``(batch, seq_len, hidden_dim)``.
        attn_weights : Optional[Tensor]
            Shape ``(batch, num_heads, seq_len, seq_len)`` if
            ``return_attn_weights=True``, else ``None``.
        present : Optional[Tuple[Tensor, Tensor]]
            The concatenated ``(k, v)`` for this layer when ``use_cache=True``,
            to be fed back on the next decoding step. ``None`` otherwise.
        """
        B, T, _ = x.shape
        # Number of tokens already in the cache == absolute position of x[0].
        past_len = kv_cache[0].size(2) if kv_cache is not None else 0

        # ------------------------------------------------------------------ #
        # 1. Linear projections                                               #
        # ------------------------------------------------------------------ #
        q = self.q_proj(x)  # (B, T, hidden_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ------------------------------------------------------------------ #
        # 2. Reshape to (B, H, T, head_dim) for multi-head computation       #
        # ------------------------------------------------------------------ #
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ------------------------------------------------------------------ #
        # 3. Apply Rotary Position Embeddings to Q and K                     #
        # ------------------------------------------------------------------ #
        # offset=past_len so a cached decode step rotates the new token by its
        # TRUE absolute position rather than position 0.
        q, k = apply_rope(q, k, self.rope, offset=past_len)

        # ------------------------------------------------------------------ #
        # 3b. Prepend the cache. RoPE is applied to the new k BEFORE the
        #     concat, and cached keys were already rotated when they were first
        #     computed -- so each key keeps the rotation for its own position.
        # ------------------------------------------------------------------ #
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        present = (k, v) if use_cache else None
        S = k.size(2)   # total key length (past + current)

        # ------------------------------------------------------------------ #
        # 4. Build the additive key-padding bias (if any).                    #
        #    Shape broadcasts over heads and query positions: (B, 1, 1, T).   #
        # ------------------------------------------------------------------ #
        pad_bias: Optional[Tensor] = None
        if attention_mask is not None:
            # 0 where padding → -inf added to those key columns.
            pad = (attention_mask == 0)[:, None, None, :]  # (B,1,1,S) bool
            pad_bias = torch.zeros(
                (B, 1, 1, pad.size(-1)), dtype=q.dtype, device=q.device
            ).masked_fill(pad, float("-inf"))

        if not return_attn_weights:
            # Fused, memory-efficient path (FlashAttention when available).
            # is_causal=True applies the causal mask without materialising it.
            if pad_bias is None and past_len == 0:
                context = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=True,
                    dropout_p=self.attn_dropout_p if self.training else 0.0,
                )
            elif pad_bias is None and T == 1:
                # Single-token decode: every cached key is in the past, so the
                # causal constraint is already satisfied and no mask is needed.
                context = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0,
                )
            else:
                # Combine causal + padding into one additive float mask.
                # Query i sits at absolute position past_len + i and may attend
                # to keys 0..past_len+i, so the triangle is offset by past_len.
                causal = _causal_bias(T, S, past_len, q.dtype, q.device)
                attn_bias = causal[None, None, :, :]
                if pad_bias is not None:
                    attn_bias = attn_bias + pad_bias   # (B,1,T,S)
                context = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=self.attn_dropout_p if self.training else 0.0,
                )
            attn_weights = None
        else:
            # Explicit path — needed only when the caller wants the weights.
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,S)
            causal = _causal_bias(T, S, past_len, scores.dtype, scores.device)
            scores = scores + causal[None, None, :, :]
            if pad_bias is not None:
                scores = scores + pad_bias
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
            if self.attn_dropout_p > 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.attn_dropout_p)
            attn_weights = attn_weights.to(v.dtype)
            context = torch.matmul(attn_weights, v)  # (B,H,T,head_dim)

        # ------------------------------------------------------------------ #
        # 8. Merge heads: (B, H, T, D) → (B, T, H*D) = (B, T, hidden_dim)  #
        # ------------------------------------------------------------------ #
        context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

        # ------------------------------------------------------------------ #
        # 9. Output projection                                                #
        # ------------------------------------------------------------------ #
        output = self.o_proj(context)

        return output, attn_weights, present

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )
