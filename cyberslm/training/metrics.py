"""
Training Metrics
================
Lightweight dataclass and printer for all training-time scalars.
Kept separate so the Trainer stays focused on the training loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class StepMetrics:
    """Metrics captured at a single optimizer step."""
    step:            int
    train_loss:      float
    val_loss:        Optional[float]
    learning_rate:   float
    grad_norm:       float
    tokens_per_sec:  float
    tokens_total:    int
    elapsed_sec:     float
    eta_sec:         Optional[float]
    gpu_mem_gb:      Optional[float]
    cpu_mem_gb:      Optional[float]

    def __str__(self) -> str:
        parts = [
            f"step={self.step:>6d}",
            f"loss={self.train_loss:.4f}",
        ]
        if self.val_loss is not None:
            parts.append(f"val={self.val_loss:.4f}")
        parts += [
            f"lr={self.learning_rate:.2e}",
            f"gnorm={self.grad_norm:.3f}",
            f"tok/s={self.tokens_per_sec:,.0f}",
            f"tok={self.tokens_total/1e6:.1f}M",
        ]
        if self.eta_sec is not None:
            parts.append(f"eta={_fmt_time(self.eta_sec)}")
        if self.gpu_mem_gb is not None:
            parts.append(f"gpu={self.gpu_mem_gb:.1f}GB")
        if self.cpu_mem_gb is not None:
            parts.append(f"cpu={self.cpu_mem_gb:.1f}GB")
        return "  ".join(parts)


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


def get_gpu_mem_gb() -> Optional[float]:
    """Return current GPU memory allocated in GB, or None."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return None


def get_cpu_mem_gb() -> Optional[float]:
    """Return current process RSS in GB, or None if psutil unavailable."""
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 ** 3
    except ImportError:
        return None
