"""
loss.py
=======
Cross-entropy with an auxiliary z-loss.

Cross-entropy is invariant to adding a constant to every logit, so nothing stops
the logit vector from drifting to a large magnitude during training. Once it
does, the softmax saturates, bf16 loses precision on the difference between
large numbers, and the run develops loss spikes.

z-loss penalizes that free parameter directly:

    L = CE + c * mean( logsumexp(logits)^2 )

It pins the log-partition function near zero without constraining the *relative*
logits that actually carry the prediction. c = 1e-4 is the standard value; it
contributes almost nothing to the reported loss while removing the drift.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_with_z_loss(
    logits: Tensor,
    targets: Tensor,
    z_loss_coef: float = 1e-4,
    ignore_index: int = -100,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Parameters
    ----------
    logits  : (B, T, V) or (N, V)
    targets : (B, T) or (N,)

    Returns (total_loss, ce_loss, z_loss). Only ``total_loss`` should be
    backpropagated; the other two are for logging.
    """
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

    # float32 for the softmax: this is the one place where bf16 genuinely hurts
    logits = logits.float()

    ce = F.cross_entropy(logits, targets, ignore_index=ignore_index)

    if z_loss_coef <= 0.0:
        zero = torch.zeros((), device=logits.device)
        return ce, ce.detach(), zero

    valid = targets != ignore_index
    if valid.any():
        lse = torch.logsumexp(logits[valid], dim=-1)
        z = (lse ** 2).mean()
    else:
        z = torch.zeros((), device=logits.device)

    return ce + z_loss_coef * z, ce.detach(), z.detach()


@torch.no_grad()
def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> float:
    """Next-token top-1 accuracy over non-ignored positions."""
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
    valid = targets != ignore_index
    if not valid.any():
        return 0.0
    pred = logits[valid].argmax(dim=-1)
    return (pred == targets[valid]).float().mean().item()
