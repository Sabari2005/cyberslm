"""
schedule.py
===========
Warmup-Stable-Decay (WSD) learning-rate schedule.

    |  warmup  |            stable            |   decay   |
    0 -> peak        peak (constant)          peak -> min

Why not cosine
--------------
Cosine has to know the total step count up front, and its intermediate
checkpoints are all mid-decay, so a run stopped early yields a model that was
never annealed. WSD holds the peak rate for most of training and only anneals at
the end, which gives two practical advantages on a limited budget:

  * You can extend a run: keep training from the end of the stable phase and
    re-anneal, without restarting.
  * The stable-phase checkpoint is a legitimate starting point for a *different*
    decay length, so one expensive stable phase can produce several models.

The sharp loss drop during the decay phase is expected, not a bug.
"""

from __future__ import annotations

import math
from typing import Literal


class WSDSchedule:
    """Multiplier in [min_lr_frac, 1.0] applied to each group's base LR."""

    def __init__(
        self,
        max_steps: int,
        warmup_steps: int = 500,
        decay_frac: float = 0.2,
        min_lr_frac: float = 0.0,
        decay_shape: Literal["linear", "sqrt", "cosine"] = "linear",
    ) -> None:
        if warmup_steps < 0 or max_steps <= 0:
            raise ValueError("max_steps must be > 0 and warmup_steps >= 0")
        if not 0.0 <= decay_frac < 1.0:
            raise ValueError(f"decay_frac must be in [0, 1), got {decay_frac}")
        if warmup_steps >= max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < max_steps ({max_steps})"
            )

        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr_frac = min_lr_frac
        self.decay_shape = decay_shape
        self.decay_steps = int(decay_frac * max_steps)
        self.decay_start = max_steps - self.decay_steps

    def multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            # linear warmup; step+1 so the very first step is non-zero
            return (step + 1) / max(1, self.warmup_steps)

        if step < self.decay_start or self.decay_steps == 0:
            return 1.0

        progress = (step - self.decay_start) / max(1, self.decay_steps)
        progress = min(1.0, max(0.0, progress))

        if self.decay_shape == "linear":
            factor = 1.0 - progress
        elif self.decay_shape == "sqrt":
            factor = 1.0 - math.sqrt(progress)
        else:  # cosine
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return self.min_lr_frac + (1.0 - self.min_lr_frac) * factor


class CosineSchedule:
    """Classic linear warmup into cosine decay, kept for comparison runs."""

    def __init__(
        self,
        max_steps: int,
        warmup_steps: int = 500,
        min_lr_frac: float = 0.1,
    ) -> None:
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr_frac = min_lr_frac

    def multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_frac + (1.0 - self.min_lr_frac) * cos


def build_schedule(cfg):
    if cfg.schedule == "wsd":
        return WSDSchedule(
            max_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            decay_frac=cfg.decay_frac,
            min_lr_frac=cfg.min_lr_frac,
        )
    return CosineSchedule(
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        min_lr_frac=max(cfg.min_lr_frac, 0.1),
    )


def apply_lr(optimizer, schedule, step: int) -> float:
    """
    Scale every group's LR by the schedule multiplier.

    Base LRs are captured on the first call so repeated application does not
    compound. Muon and AdamW groups have different base rates, and both are
    scaled by the same multiplier.
    """
    mult = schedule.multiplier(step)
    for group in optimizer.param_groups:
        if "_base_lr" not in group:
            group["_base_lr"] = group.get("lr", 0.0)
            group["_base_adamw_lr"] = group.get("adamw_lr", group.get("lr", 0.0))
        group["lr"] = group["_base_lr"] * mult
        if "adamw_lr" in group:
            group["adamw_lr"] = group["_base_adamw_lr"] * mult
    return mult
