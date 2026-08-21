"""
rope.py
=======
Rotary Position Embedding (Su et al., 2021).

RoPE encodes absolute position by rotating each 2-dimensional slice of a
query/key vector by an angle proportional to its position. Because a dot
product between two rotated vectors depends only on the *difference* of their
rotation angles, attention scores become a function of relative distance:

    <R(m)q, R(n)k> = <q, R(n-m)k>

That identity is the whole point: no learned position table, and the model
extrapolates far more gracefully than with absolute embeddings.

Frequencies:  theta_i = base^(-2i/head_dim),  i = 0 .. head_dim/2 - 1

Long-context extension is supported via ``scaling`` (linear position
interpolation, Chen et al. 2023): dividing positions by a factor s lets a model
trained at length L operate at length s*L after a short adaptation phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    """Map (x1, x2) -> (-x2, x1) over the two halves of the last dimension."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Precomputes cos/sin tables and applies the rotation to q and k.

    Tables are registered as non-persistent buffers: they are pure functions of
    (head_dim, base, scaling) so they are rebuilt on load rather than bloating
    every checkpoint.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        scaling: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.base = base
        self.scaling = scaling
        self.max_seq_len = max_seq_len

        cos, sin = self._build_tables(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_tables(self, seq_len: int) -> tuple[Tensor, Tensor]:
        # theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        if self.scaling != 1.0:
            t = t / self.scaling                      # linear position interpolation
        freqs = torch.outer(t, inv_freq)              # (T, d/2)
        # duplicate so the table lines up with the (x1 | x2) split in _rotate_half
        emb = torch.cat((freqs, freqs), dim=-1)       # (T, d)
        return emb.cos(), emb.sin()

    def _tables_for(self, seq_len: int, device, dtype) -> tuple[Tensor, Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            # grow on demand (e.g. during long-context adaptation)
            cos, sin = self._build_tables(seq_len)
            self.cos_cached = cos.to(device)
            self.sin_cached = sin.to(device)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """
        Rotate q and k.

        Shapes: q is (B, n_heads, T, head_dim), k is (B, n_kv_heads, T, head_dim).
        ``offset`` is the number of cached tokens already generated, so that
        incremental decoding rotates by the true absolute position.
        """
        seq_len = q.shape[-2]
        cos, sin = self._tables_for(offset + seq_len, q.device, q.dtype)
        cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,T,d)
        sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, base={self.base}, "
            f"scaling={self.scaling}, max_seq_len={self.max_seq_len}"
        )
