"""
RMSNorm — Root Mean Square Layer Normalisation
===============================================
Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
           https://arxiv.org/abs/1910.07467

Mathematical definition
-----------------------
Given an input vector **x** ∈ ℝ^d:

    RMS(x) = sqrt( (1/d) * Σ xᵢ² + ε )

    RMSNorm(x) = (x / RMS(x)) * γ

where γ ∈ ℝ^d is a learned per-channel scale (initialised to 1.0) and
ε > 0 is a small constant for numerical stability.

Key differences from LayerNorm
-------------------------------
- No mean subtraction (no re-centering step).
- No learned bias β (the bias-free variant).
- ~30 % fewer operations than LayerNorm, which matters across 12 layers.
- Empirically matches or exceeds LayerNorm in modern transformer training.

Numerical stability
-------------------
- The RMS is computed in float32 regardless of input dtype. This prevents
  underflow/overflow when activations are in bfloat16 or float16.  For our
  FP32 training runs this cast is a no-op but it is correct and future-proof.
- ε = 1e-6 (default) prevents division by zero even for near-zero inputs.
- The scale γ is cast back to the input dtype before multiplication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation without mean-centering or bias.

    Parameters
    ----------
    dim : int
        Feature dimension to normalise over (last dimension of the input).
    eps : float
        Small constant added to the RMS denominator for numerical stability.
        Defaults to 1e-6.

    Shape
    -----
    Input  : ``(*, dim)``  — any leading batch / sequence dimensions.
    Output : ``(*, dim)``  — same shape as input.

    Examples
    --------
    >>> norm = RMSNorm(384, eps=1e-6)
    >>> x = torch.randn(2, 512, 384)
    >>> y = norm(x)
    >>> y.shape
    torch.Size([2, 512, 384])
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.eps = eps
        # Learned per-channel scale, initialised to 1 (identity transform).
        self.weight = nn.Parameter(torch.ones(dim))

    def _compute_rms(self, x: Tensor) -> Tensor:
        """
        Compute the RMS over the last dimension.

        Always promotes to float32 to prevent numerical issues with
        reduced-precision dtypes. For FP32 training this is a no-op.

        Returns
        -------
        Tensor
            Shape ``(*, 1)`` — one RMS value per token position.
        """
        return x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` by its per-token root mean square.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(*, dim)``.

        Returns
        -------
        Tensor
            Normalised output of the same shape and dtype as ``x``.
        """
        rms = self._compute_rms(x)
        # Normalise in float32, then cast back to original dtype.
        x_normed = x.float() / rms
        # Scale by learned weights (cast to input dtype for type safety).
        return (x_normed * self.weight.float()).to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
