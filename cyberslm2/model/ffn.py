"""
ffn.py
======
SwiGLU feed-forward network (Shazeer, 2020).

    SwiGLU(x) = W_down( SiLU(W_gate x) * W_up x )

The elementwise product gives the layer a multiplicative interaction that a
plain MLP lacks, which is worth roughly a 1-2% loss improvement at equal
parameter count. Because it uses three matrices instead of two, the inner width
is set to (8/3)*d_model rather than 4*d_model so the parameter budget matches.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    """Gated feed-forward block. All projections are bias-free."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
