"""
block.py
========
Pre-norm decoder block.

    h = x + Attn(RMSNorm(x))
    y = h + FFN(RMSNorm(h))

Pre-norm (normalizing the *input* of each sublayer rather than the output) keeps
an unobstructed identity path from the embedding to the final norm, so gradients
reach early layers without vanishing. Post-norm transformers need careful warmup
and often diverge at this depth without it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from cyberslm2.configs.presets import ModelConfig
from cyberslm2.model.attention import GroupedQueryAttention
from cyberslm2.model.ffn import SwiGLU
from cyberslm2.model.norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        attn_out, new_cache = self.attn(
            self.attn_norm(x), attn_mask=attn_mask, kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache
