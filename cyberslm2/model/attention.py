"""
attention.py
============
Grouped-Query Attention (Ainslie et al., 2023) with RoPE, optional QK-norm,
document-aware masking and an incremental KV cache.

Why GQA
-------
Multi-head attention stores one K and one V per query head. At inference the KV
cache is the dominant memory cost and it scales with n_heads. GQA lets several
query heads share a single KV head, cutting cache size by n_heads/n_kv_heads
with almost no quality loss.

For this model: 8 query heads share 2 KV heads (a 4x reduction). Concretely the
50M config caches 2*12*2048*128*2 bytes = 12.6 MB at full context instead of
50.3 MB under MHA — which is what makes long agentic traces affordable.

Quality note: GQA is not free, but the loss it costs is far smaller than the
loss from having to shrink the context or batch to fit an MHA cache.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cyberslm2.configs.presets import ModelConfig
from cyberslm2.model.norm import RMSNorm
from cyberslm2.model.rope import RotaryEmbedding

def _sdpa_supports_gqa() -> bool:
    """
    torch >= 2.5 can broadcast KV heads inside SDPA (enable_gqa), which avoids
    materializing the expanded K/V. Detected by version because SDPA is a
    C-bound builtin whose signature cannot be introspected reliably.
    """
    try:
        major, minor = (int(p) for p in torch.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        return False
    return (major, minor) >= (2, 5)


_SDPA_HAS_GQA = _sdpa_supports_gqa()


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """
    Expand KV heads to match query heads: (B, n_kv, T, D) -> (B, n_kv*n_rep, T, D).

    Uses expand (a view) before reshape so no data is copied until the reshape
    forces it, which keeps this cheap.
    """
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, t, d)
        .reshape(b, n_kv * n_rep, t, d)
    )


class GroupedQueryAttention(nn.Module):
    """Self-attention with GQA, RoPE, and an optional KV cache."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_kv_groups
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        d, hd = config.d_model, config.head_dim
        self.wq = nn.Linear(d, self.n_heads * hd, bias=False)
        self.wk = nn.Linear(d, self.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(d, self.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(self.n_heads * hd, d, bias=False)

        # QK-norm keeps attention logits from drifting to extreme magnitudes,
        # which is the most common cause of loss spikes in small models trained
        # at high learning rate.
        if config.qk_norm:
            self.q_norm: nn.Module = RMSNorm(hd, config.norm_eps)
            self.k_norm: nn.Module = RMSNorm(hd, config.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rope = RotaryEmbedding(
            head_dim=hd,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            scaling=config.rope_scaling,
        )

        self.attn_dropout = config.attn_dropout

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Parameters
        ----------
        x         : (B, T, d_model)
        attn_mask : optional boolean mask (B, 1, T, S) where True == "may attend".
                    When None, plain causal masking is used (the fast path).
                    The data pipeline supplies this to prevent packed documents
                    from attending across each other's boundaries.
        kv_cache  : optional (k, v) from previous decoding steps
        use_cache : return the updated cache for incremental generation
        """
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotate by absolute position: with a cache, this block starts at
        # however many tokens were already generated.
        offset = kv_cache[0].shape[-2] if kv_cache is not None else 0
        q, k = self.rope(q, k, offset=offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=-2)
            v = torch.cat([kv_cache[1], v], dim=-2)
        new_cache = (k, v) if use_cache else None

        dropout_p = self.attn_dropout if self.training else 0.0

        # Causal only makes sense when q and k are the same length; during
        # cached decoding a single query token must see the whole cache.
        is_causal = attn_mask is None and offset == 0 and T > 1

        if _SDPA_HAS_GQA:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=True,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out), new_cache
