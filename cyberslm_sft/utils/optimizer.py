"""
CyberSLM SFT — Optimizer & Scheduler
======================================
Factory functions for AdamW and cosine-with-warmup LR schedule.

Design decisions
----------------
* Weight decay is applied only to non-bias, non-norm parameters (standard
  practice; applying it to embeddings and LayerNorm scales is harmful).
* Learning rate is intentionally much lower than pretraining (2e-5 default)
  because SFT is *refinement*, not relearning.
* Cosine decay with a linear warmup phase and a configurable minimum LR
  floor (``min_lr_ratio``) prevents the LR from decaying all the way to
  zero, which can cause instability near the end of training.
* A linear and constant schedule are also supported for ablations.
"""

from __future__ import annotations

import logging
import math
from typing import List, Set

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from configs.sft_config import TrainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------

def _get_param_groups(
    model:        nn.Module,
    weight_decay: float,
) -> List[dict]:
    """
    Split parameters into two groups:
    * ``decay``   — weight matrices, embeddings → apply weight decay
    * ``no_decay`` — biases, LayerNorm/RMSNorm weights → no weight decay

    Parameters
    ----------
    model:        The model whose parameters to group.
    weight_decay: Decay coefficient applied to the ``decay`` group.

    Returns
    -------
    List of two param-group dicts compatible with ``torch.optim``.
    """
    decay:    List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []

    # Track object ids to avoid double-counting shared weights
    seen: Set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        # Biases, 1-D tensors (norms, scales), and the (tied) token embedding
        # skip weight decay. The embedding shares storage with the LM head, so
        # decaying it also shrinks the output projection / logit scale.
        is_embedding = ("embedding" in name) or name.endswith("lm_head.weight")
        if param.ndim == 1 or name.endswith(".bias") or is_embedding:
            no_decay.append(param)
        else:
            decay.append(param)

    logger.debug(
        "Param groups — decay: %d tensors | no_decay: %d tensors",
        len(decay), len(no_decay),
    )

    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(
    model:      nn.Module,
    cfg:        TrainConfig,
) -> optim.AdamW:
    """
    Build an AdamW optimizer with correct parameter grouping.

    Parameters
    ----------
    model:  The instruction-tuned model.
    cfg:    ``TrainConfig`` for LR, betas, eps, and weight decay.

    Returns
    -------
    Configured ``torch.optim.AdamW`` instance.
    """
    param_groups = _get_param_groups(model, weight_decay=cfg.weight_decay)

    optimizer = optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=False,  # keep FP32-safe; fused AdamW requires CUDA
    )

    n_decay    = len(param_groups[0]["params"])
    n_no_decay = len(param_groups[1]["params"])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "AdamW — lr=%.2e | wd=%.4f | β=(%.2f, %.4f) | "
        "decay params: %d | no-decay params: %d | total trainable: %d",
        cfg.learning_rate, cfg.weight_decay,
        cfg.beta1, cfg.beta2,
        n_decay, n_no_decay, total_params,
    )

    return optimizer


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer:    optim.Optimizer,
    cfg:          TrainConfig,
    total_steps:  int,
) -> LambdaLR:
    """
    Build a LambdaLR scheduler implementing linear warmup followed by the
    chosen decay strategy.

    The LR at step *t* is::

        lr(t) = base_lr × lambda(t)

    where ``lambda(t)`` is computed by one of the three strategies below.

    Cosine (default)::

        warmup:  t / warmup_steps
        decay:   min_lr + 0.5 × (1 - min_lr) × (1 + cos(π × progress))

    Linear::

        warmup:  t / warmup_steps
        decay:   max(min_lr, 1 - (t - warmup_steps) / (total - warmup_steps))

    Constant::

        warmup:  t / warmup_steps
        plateau: 1.0  (constant after warmup)

    Parameters
    ----------
    optimizer:   The AdamW optimizer to attach the schedule to.
    cfg:         ``TrainConfig`` for warmup_ratio, lr_schedule, min_lr_ratio.
    total_steps: Total number of gradient *update* steps (not forward passes).

    Returns
    -------
    ``torch.optim.lr_scheduler.LambdaLR``
    """
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    min_lr_mult  = cfg.min_lr_ratio   # lambda floor

    logger.info(
        "Scheduler: %s | total steps: %d | warmup steps: %d (%.1f %%) | "
        "min_lr_ratio: %.3f",
        cfg.lr_schedule, total_steps, warmup_steps,
        100.0 * warmup_steps / max(total_steps, 1),
        min_lr_mult,
    )

    schedule = cfg.lr_schedule.lower()

    if schedule == "cosine":
        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return max(1e-8, float(current_step) / float(warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_mult + (1.0 - min_lr_mult) * cosine_decay

    elif schedule == "linear":
        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return max(1e-8, float(current_step) / float(warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(min_lr_mult, 1.0 - progress)

    elif schedule == "constant":
        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return max(1e-8, float(current_step) / float(warmup_steps))
            return 1.0

    else:
        raise ValueError(
            f"Unknown lr_schedule '{cfg.lr_schedule}'. "
            "Choose from: 'cosine', 'linear', 'constant'."
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Gradient clipping helper
# ---------------------------------------------------------------------------

def clip_gradients(
    model:          nn.Module,
    max_grad_norm:  float,
) -> float:
    """
    Clip gradients in-place and return the pre-clip global gradient norm.

    Parameters
    ----------
    model:         Model with ``requires_grad`` parameters.
    max_grad_norm: Maximum norm threshold (e.g. 1.0).

    Returns
    -------
    float — the unclipped gradient norm (for logging).
    """
    grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    return float(grad_norm)
