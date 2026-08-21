"""
norm.py
=======
RMSNorm — root-mean-square layer normalization (Zhang & Sennrich, 2019).

Drops the mean-centering of LayerNorm and keeps only the rescaling:

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g

This is cheaper (no mean pass) and empirically matches LayerNorm quality in
transformers, which is why every LLaMA-class model uses it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel gain."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        # rsqrt in float32: the mean of squares underflows badly in fp16/bf16
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"
