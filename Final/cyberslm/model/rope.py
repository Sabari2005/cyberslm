"""
Rotary Position Embedding (RoPE)
=================================
Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           (Su et al., 2021) — https://arxiv.org/abs/2104.09864

Mathematical definition
-----------------------
For a query (or key) vector **q** at position ``m`` with head dimension ``d``:

  1. Partition **q** into pairs: (q₁, q₂), (q₃, q₄), ..., (q_{d-1}, q_d).

  2. For each pair index ``i ∈ {0, 1, ..., d/2 - 1}`` define the frequency:

         θᵢ = 1 / base^(2i / d)         (base = 10 000)

  3. Apply a 2-D rotation to each pair at position ``m``:

         R(m, θᵢ) · (q_{2i}, q_{2i+1}) =
             (q_{2i} cos(m·θᵢ) − q_{2i+1} sin(m·θᵢ),
              q_{2i} sin(m·θᵢ) + q_{2i+1} cos(m·θᵢ))

This is equivalent to multiplying **q** (viewed as complex numbers) by
``exp(i · m · θ)``, which preserves the inner product of relative positions:

    ⟨R(m)q, R(n)k⟩ depends only on (m − n),

giving translation-equivariant attention without absolute position tokens.

Efficient implementation
------------------------
The rotation can be expressed without complex arithmetic:

    q_rot = [q_even · cos − q_odd · sin,
             q_even · sin + q_odd · cos]

where ``q_even = q[..., 0::2]``, ``q_odd = q[..., 1::2]``.

Numerically interleaved form (even/odd) vs. split-half form (first/second half)
are equivalent; we use the interleaved form for clarity.

Precomputation
--------------
``cos`` and ``sin`` tensors of shape ``(max_seq_len, head_dim // 2)`` are
computed once and registered as non-parameter buffers so they move with the
module (CPU ↔ GPU) and are not included in ``state_dict`` checkpoints.

Stability notes
---------------
- Frequencies are computed in float64 then cast to float32 to minimise
  floating-point error in ``pow`` and ``arange``.
- All rotations are executed in float32 to prevent loss of precision.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionEmbedding(nn.Module):
    """
    Precomputed Rotary Position Embedding cache.

    Registers ``cos`` and ``sin`` buffers of shape
    ``(max_seq_len, head_dim // 2)`` at construction time. Applying RoPE
    to a query or key tensor costs only element-wise multiplications and
    additions — no matrix multiplications.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head.  Must be even.
    max_seq_len : int
        Maximum sequence length to pre-compute.  Sequences longer than this
        will raise an error at runtime.
    base : int
        RoPE base frequency (10 000 in the original paper).

    Shape of ``apply``
    ------------------
    Input  : ``(batch, num_heads, seq_len, head_dim)``
    Output : ``(batch, num_heads, seq_len, head_dim)``
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: int = 10_000,
    ) -> None:
        super().__init__()

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be a positive even integer, got {head_dim}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute and register buffers (not model parameters).
        cos_cache, sin_cache = self._build_cache(head_dim, max_seq_len, base)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    @staticmethod
    def _build_cache(
        head_dim: int,
        max_seq_len: int,
        base: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Build ``(cos, sin)`` caches of shape ``(max_seq_len, head_dim // 2)``.

        Computation is performed in float64 for precision, then cast to
        float32 for storage.

        Returns
        -------
        Tuple[Tensor, Tensor]
            ``cos_cache`` and ``sin_cache``, each of shape
            ``(max_seq_len, head_dim // 2)``.
        """
        half_dim = head_dim // 2

        # θᵢ = 1 / base^(2i / head_dim)  for i ∈ {0, ..., half_dim - 1}
        # Computed in float64 to avoid precision loss in the exponent.
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        )
        # Shape: (half_dim,)

        # Position indices m ∈ {0, 1, ..., max_seq_len - 1}
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        # Shape: (max_seq_len,)

        # Outer product: angles[m, i] = m * θᵢ
        angles = torch.outer(positions, inv_freq)
        # Shape: (max_seq_len, half_dim)

        cos_cache = angles.cos().to(torch.float32)
        sin_cache = angles.sin().to(torch.float32)

        return cos_cache, sin_cache

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """
        Rotate the last dimension by interleaving even/odd pairs.

        For input ``x`` of shape ``(..., head_dim)``:

            x_even = x[..., 0::2]  (positions 0, 2, 4, ...)
            x_odd  = x[..., 1::2]  (positions 1, 3, 5, ...)

        Returns ``[-x_odd, x_even]`` interleaved back into ``(..., head_dim)``.
        This is the standard rotation that implements the complex-number trick.

        Parameters
        ----------
        x : Tensor
            Shape ``(..., head_dim)`` where ``head_dim`` is even.

        Returns
        -------
        Tensor
            Same shape as ``x``.
        """
        x_even = x[..., 0::2]  # (..., head_dim // 2)
        x_odd = x[..., 1::2]   # (..., head_dim // 2)
        # Interleave: stack along new dim then flatten.
        rotated = torch.stack([-x_odd, x_even], dim=-1)
        # Shape: (..., head_dim // 2, 2) → (..., head_dim)
        return rotated.flatten(start_dim=-2)

    def apply(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Apply Rotary Position Embeddings to ``x``.

        Parameters
        ----------
        x : Tensor
            Query or key tensor of shape
            ``(batch, num_heads, seq_len, head_dim)``.
        offset : int
            Absolute position of ``x[..., 0, :]`` in the full sequence.

            This is what makes incremental decoding correct. With a KV cache the
            model feeds one token at a time, so ``seq_len == 1`` and the naive
            ``cos_cache[:1]`` would rotate every generated token as if it were at
            position 0 -- destroying all positional information after the prompt.
            Passing ``offset=len(cache)`` selects the true absolute position.

        Returns
        -------
        Tensor
            Rotated tensor with the same shape and dtype as ``x``.

        Raises
        ------
        ValueError
            If ``seq_len`` exceeds ``max_seq_len``.
        """
        seq_len = x.size(2)
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"Positions [{offset}, {offset + seq_len}) exceed RoPE cache size "
                f"{self.max_seq_len}.  Re-instantiate with a larger max_seq_len."
            )

        # Retrieve cached values for this absolute position span.
        # cos_cache: (seq_len, head_dim // 2)
        # sin_cache: (seq_len, head_dim // 2)
        cos = self.cos_cache[offset : offset + seq_len]  # type: ignore[index]
        sin = self.sin_cache[offset : offset + seq_len]  # type: ignore[index]

        # Expand to broadcast over batch and head dimensions:
        # (1, 1, seq_len, head_dim // 2) → broadcasts with (B, H, T, D/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Interleave cos and sin to match full head_dim.
        # Each of (cos, sin) has shape (1, 1, T, D/2).
        # We need (1, 1, T, D) by interleaving even positions with cos,
        # odd positions with sin.  The _rotate_half trick handles this:
        #
        #   x_rot = x * cos_full + rotate_half(x) * sin_full
        #
        # where cos_full[..., 0::2] = cos and cos_full[..., 1::2] = cos,
        # i.e. each cos value applies to both the even AND its paired odd slot.
        # Achieved by repeating each half-dim value into both slots.
        cos_full = cos.repeat_interleave(2, dim=-1)  # (1, 1, T, D)
        sin_full = sin.repeat_interleave(2, dim=-1)  # (1, 1, T, D)

        # Work in float32 for stability, then restore original dtype.
        x_fp32 = x.float()
        x_rot = x_fp32 * cos_full + self._rotate_half(x_fp32) * sin_full
        return x_rot.to(x.dtype)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Alias for :meth:`apply` to support ``nn.Sequential`` usage."""
        return self.apply(x, offset=offset)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"base={self.base}"
        )


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------

def apply_rope(
    q: Tensor,
    k: Tensor,
    rope: RotaryPositionEmbedding,
    offset: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Apply the same RoPE instance to both query and key tensors.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
    k : Tensor
        Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
    rope : RotaryPositionEmbedding
        Pre-built RoPE module (carries the cos/sin cache on the correct device).

    Returns
    -------
    Tuple[Tensor, Tensor]
        ``(q_rot, k_rot)`` — rotated queries and keys.
    """
    return rope.apply(q, offset=offset), rope.apply(k, offset=offset)
