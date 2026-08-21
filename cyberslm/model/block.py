"""
Transformer Decoder Block
==========================
A single Pre-Norm residual decoder block consisting of:
  1. RMSNorm → Multi-Head Self Attention → residual add
  2. RMSNorm → SwiGLU FFN               → residual add

Pre-Norm architecture
---------------------
Post-Norm (original Transformer):
    x = LayerNorm(x + SubLayer(x))

Pre-Norm (modern: GPT-2 onward, LLaMA, etc.):
    x = x + SubLayer(LayerNorm(x))

Pre-Norm is strongly preferred for deep networks because:
  - Gradients flow through the residual connection bypassing the
    normalisation, preventing vanishing gradients in very deep stacks.
  - Training is more stable without learning-rate warmup tricks.
  - Final RMSNorm on the output is added at the model level (not here)
    to normalise the final residual stream before the LM head.

Residual stream
---------------
The residual stream x ∈ ℝ^{B×T×d} is the backbone of the model. Each
sub-layer reads from it, computes a delta, and adds back:

    Δ_attn = MHSA( RMSNorm(x) )
    x       = x + Δ_attn

    Δ_ffn  = FFN( RMSNorm(x) )
    x       = x + Δ_ffn

This additive structure means the gradient of the loss with respect to
early layers contains a direct path through the identity (residual),
enabling reliable training of 12+ layer models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from cyberslm.model.config import CyberSLMConfig
from cyberslm.model.norm import RMSNorm
from cyberslm.model.attention import MultiHeadSelfAttention
from cyberslm.model.ffn import SwiGLUFeedForward


class DecoderBlock(nn.Module):
    """
    Pre-Norm Transformer Decoder Block.

    Parameters
    ----------
    config : CyberSLMConfig
        Validated model configuration.
    layer_idx : int
        Zero-based index of this block in the stack (used for display only).

    Sub-modules
    -----------
    attn_norm : RMSNorm
        Normalises the residual stream before attention.
    attn : MultiHeadSelfAttention
        Self attention with RoPE and causal masking.
    ffn_norm : RMSNorm
        Normalises the residual stream before the FFN.
    ffn : SwiGLUFeedForward
        SwiGLU position-wise feed-forward network.

    Shape
    -----
    Input  : ``(batch, seq_len, hidden_dim)``
    Output : ``(batch, seq_len, hidden_dim)``
    """

    def __init__(
        self,
        config: CyberSLMConfig,
        layer_idx: int = 0,
        rope=None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm before attention.
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        # Multi-head self attention (owns RoPE + causal mask buffers).
        self.attn = MultiHeadSelfAttention(config, rope=rope)

        # Pre-norm before FFN.
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        # SwiGLU feed-forward.
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """
        Apply one Pre-Norm residual decoder block.

        Parameters
        ----------
        x : Tensor
            Residual stream of shape ``(batch, seq_len, hidden_dim)``.
        attention_mask : Optional[Tensor]
            Key-padding mask ``(batch, seq_len)`` (1=keep, 0=pad), forwarded
            to the attention sub-layer. ``None`` for packed/unpadded batches.
        return_attn_weights : bool
            Propagated to the attention sub-layer.

        Returns
        -------
        x : Tensor
            Updated residual stream, same shape as input.
        attn_weights : Optional[Tensor]
            Attention weights if requested, else None.
        """
        # ---- Attention sub-layer ----------------------------------------- #
        attn_out, attn_weights, present = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + attn_out

        # ---- FFN sub-layer ------------------------------------------------ #
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights, present

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}"
