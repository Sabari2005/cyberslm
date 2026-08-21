"""
Learning Rate Scheduler — Linear Warmup + Cosine Decay
=======================================================
Two-phase schedule widely used in LLM pretraining (GPT-3, LLaMA, etc.):

Phase 1 — Linear warmup (steps 0 … warmup_steps):
    lr(t) = peak_lr × (t / warmup_steps)

Phase 2 — Cosine decay (steps warmup_steps … max_steps):
    progress = (t - warmup_steps) / (max_steps - warmup_steps)
    lr(t) = min_lr + 0.5 × (peak_lr - min_lr) × (1 + cos(π × progress))

After max_steps the schedule stays at min_lr.

Why cosine decay?
-----------------
Cosine annealing provides a smooth, monotonically decreasing schedule that
starts fast and slows down near the end.  The slow tail allows the optimizer
to settle into flat minima (good generalisation) without bouncing.

Why linear warmup?
------------------
Starting with a large learning rate from step 0 causes gradient explosion:
the model parameters are randomly initialised and the loss surface is steep.
A gradual warmup lets the model reach a stable loss basin before applying
the full learning rate.

Implementation note
-------------------
This is a functional scheduler, not an nn.Module.  We update the optimizer's
param_groups directly to avoid PyTorch scheduler state-dict quirks.
State (current step) is tracked in CheckpointManager alongside the optimizer.
"""

from __future__ import annotations

import math
from typing import List

import torch.optim as optim


class CosineWarmupScheduler:
    """
    Linear warmup followed by cosine decay.

    Manages the learning rate by directly setting ``optimizer.param_groups``
    on each call to :meth:`step`.  Compatible with gradient accumulation
    (call :meth:`step` once per *optimizer* step, not per micro-batch).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate is managed.
    peak_lr : float
        Maximum learning rate (reached at end of warmup).
    min_lr : float
        Minimum learning rate (floor of cosine decay).
    warmup_steps : int
        Number of optimizer steps for the linear warmup phase.
    max_steps : int
        Total optimizer steps (warmup + decay).
    last_step : int
        Step to resume from (used when restoring from checkpoint).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
        last_step: int = 0,
    ) -> None:
        if peak_lr <= 0:
            raise ValueError(f"peak_lr must be positive, got {peak_lr}")
        if min_lr <= 0 or min_lr > peak_lr:
            raise ValueError(
                f"min_lr must be in (0, peak_lr], got {min_lr} vs {peak_lr}"
            )
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if last_step < 0:
            raise ValueError(f"last_step must be >= 0, got {last_step}")

        self.optimizer    = optimizer
        self.peak_lr      = peak_lr
        self.min_lr       = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self._step        = last_step

        # Immediately set the LR to the correct value for the current step.
        self._set_lr(self.get_lr(self._step))

    # ---------------------------------------------------------------------- #
    # Core schedule                                                            #
    # ---------------------------------------------------------------------- #

    def get_lr(self, step: int) -> float:
        """
        Return the learning rate for a given optimizer step.

        Parameters
        ----------
        step : int
            Current optimizer step (0-indexed).

        Returns
        -------
        float
            Learning rate at that step.
        """
        if step < self.warmup_steps:
            # Linear warmup: avoid division by zero when warmup_steps == 0.
            if self.warmup_steps == 0:
                return self.peak_lr
            return self.peak_lr * (step / self.warmup_steps)

        if step >= self.max_steps:
            return self.min_lr

        # Cosine decay.
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine_factor * (self.peak_lr - self.min_lr)

    def step(self) -> float:
        """
        Advance one optimizer step and update the optimizer's learning rate.

        Returns
        -------
        float
            The new learning rate after this step.
        """
        self._step += 1
        lr = self.get_lr(self._step)
        self._set_lr(lr)
        return lr

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ---------------------------------------------------------------------- #
    # State                                                                    #
    # ---------------------------------------------------------------------- #

    @property
    def current_step(self) -> int:
        """Current optimizer step (0-indexed, before the first :meth:`step` call)."""
        return self._step

    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        return self.get_lr(self._step)

    def state_dict(self) -> dict:
        return {
            "step":         self._step,
            "peak_lr":      self.peak_lr,
            "min_lr":       self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps":    self.max_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step        = state["step"]
        self.peak_lr      = state["peak_lr"]
        self.min_lr       = state["min_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.max_steps    = state["max_steps"]
        self._set_lr(self.get_lr(self._step))
