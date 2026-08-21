"""
SwiGLU Feed-Forward Network (FFN)
===================================
Reference: "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
           https://arxiv.org/abs/2002.05202

Mathematical definition
-----------------------
Standard FFN (for contrast):
    FFN(x) = activation(x W₁) W₂

SwiGLU FFN:
    SwiGLU(x) = (x W_gate ⊙ swish(x W_gate)) W₂   ← WRONG shorthand
    
Correct form with separate gate and value projections:
    gate(x) = x W_gate        ∈ ℝ^{B×T×ffn_dim}
    val(x)  = x W_val         ∈ ℝ^{B×T×ffn_dim}
    hidden  = swish(gate(x)) ⊙ val(x)
    out     = hidden W_out    ∈ ℝ^{B×T×hidden_dim}

where swish(z) = z · sigmoid(z) = z · σ(z).

Why SwiGLU
----------
- The gating mechanism (⊙) gives the network a multiplicative path to
  control information flow — intuitively "how much" of each feature passes
  through at each position.
- swish is smooth and non-monotonic, empirically outperforming ReLU and
  GELU in large-scale experiments (PaLM, LLaMA, etc.).
- The factored W_gate / W_val structure adds one extra matrix but improves
  quality relative to a standard 2-layer FFN of the same parameter budget.

Parameter count
---------------
Three matrices: W_gate, W_val, W_out
    W_gate : hidden_dim × ffn_hidden_dim   (384 × 1024 = 393 216)
    W_val  : hidden_dim × ffn_hidden_dim   (384 × 1024 = 393 216)
    W_out  : ffn_hidden_dim × hidden_dim   (1024 × 384 = 393 216)
    Total per layer: 1 179 648 ≈ 1.18 M

No bias on any projection (consistent with modern practice).

Numerical stability
-------------------
- swish(z) = z · σ(z) is numerically safe for all real z.
- σ(z) = 1/(1+exp(−z)) in PyTorch uses a numerically stable implementation.
- The element-wise product of two bounded (after sigmoid) quantities does
  not amplify values explosively.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cyberslm.model.config import CyberSLMConfig


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Consists of three bias-free linear projections:
    ``gate_proj``, ``val_proj``, and ``out_proj``.

    Parameters
    ----------
    config : CyberSLMConfig
        Validated model configuration.

    Shape
    -----
    Input  : ``(batch, seq_len, hidden_dim)``
    Output : ``(batch, seq_len, hidden_dim)``
    """

    def __init__(self, config: CyberSLMConfig) -> None:
        super().__init__()
        self.hidden_dim     = config.hidden_dim
        self.ffn_hidden_dim = config.ffn_hidden_dim

        # Gate projection: produces the gating signal fed through swish.
        self.gate_proj = nn.Linear(
            config.hidden_dim, config.ffn_hidden_dim, bias=False
        )
        # Value projection: produces the value signal gated element-wise.
        self.val_proj = nn.Linear(
            config.hidden_dim, config.ffn_hidden_dim, bias=False
        )
        # Output projection: maps back to residual stream dimension.
        self.out_proj = nn.Linear(
            config.ffn_hidden_dim, config.hidden_dim, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU feed-forward transformation.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, seq_len, hidden_dim)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, seq_len, hidden_dim)``.
        """
        # gate:   (B, T, ffn_hidden_dim)
        # val:    (B, T, ffn_hidden_dim)
        gate = self.gate_proj(x)
        val  = self.val_proj(x)

        # SwiGLU: swish(gate) ⊙ val
        # F.silu is swish: silu(z) = z * sigmoid(z)
        hidden = F.silu(gate) * val  # (B, T, ffn_hidden_dim)

        # Project back to hidden_dim
        return self.out_proj(hidden)  # (B, T, hidden_dim)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"ffn_hidden_dim={self.ffn_hidden_dim}"
        )
