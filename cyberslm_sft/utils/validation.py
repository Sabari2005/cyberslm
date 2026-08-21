"""
CyberSLM SFT — Validation
==========================
Runs the validation loop and returns a structured ``ValidationResult``.

The validator is intentionally separated from the trainer so it can be
called from a standalone evaluation script as well as mid-training.

Metrics computed
----------------
* ``val_loss``     — mean cross-entropy over all non-masked response tokens
* ``val_ppl``      — perplexity (exp(val_loss))
* ``tokens_seen``  — total active (non-masked) tokens evaluated
* ``tok_per_sec``  — throughput over the validation set
* ``n_batches``    — number of batches processed
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.loss_masking import masked_cross_entropy, count_response_tokens
from data.prompt_formatter import IGNORE_INDEX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    val_loss:    float
    val_ppl:     float
    tok_per_sec: float
    tokens_seen: int
    n_batches:   int

    def __str__(self) -> str:
        return (
            f"loss={self.val_loss:.4f}  ppl={self.val_ppl:.2f}  "
            f"tok/s={int(self.tok_per_sec)}  batches={self.n_batches}"
        )


# ---------------------------------------------------------------------------
# Validation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model:      nn.Module,
    val_loader: DataLoader,
    device:     torch.device,
    max_batches: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> ValidationResult:
    """
    Run inference over the validation DataLoader and return metrics.

    Parameters
    ----------
    model:       The model to evaluate (set to eval mode internally).
    val_loader:  DataLoader yielding ``{input_ids, labels, attention_mask}``.
    device:      Target device (CPU or CUDA).
    max_batches: If set, evaluate only the first N batches (fast sanity check).

    Returns
    -------
    ``ValidationResult`` with loss, perplexity, throughput, and counts.
    """
    model.eval()

    total_loss:   float = 0.0
    total_active: int   = 0
    total_tokens: int   = 0
    n_batches:    int   = 0

    t_start = time.time()

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = _forward(model, input_ids, attention_mask)

        # Count targets AFTER the causal shift. masked_cross_entropy scores
        # labels[:, 1:], so counting the unshifted labels (the old behaviour)
        # could disagree with the numerator and skew the reported loss.
        n_active = int((labels[:, 1:] != IGNORE_INDEX).sum().item())

        if n_active == 0:
            # Nothing to evaluate in this batch — skip silently
            continue

        # Sum loss (not mean) so we can average over all tokens, not batches
        batch_loss = masked_cross_entropy(logits, labels, reduction="sum")

        total_loss   += batch_loss.item()
        total_active += n_active
        total_tokens += labels.numel()
        n_batches    += 1

    elapsed = time.time() - t_start

    if total_active == 0:
        logger.warning(
            "Validation produced zero active tokens — check your dataset "
            "and prompt formatter."
        )
        # Restore train mode on this path too; the early return used to leave
        # the model stuck in eval().
        model.train()
        return ValidationResult(
            val_loss=float("inf"),
            val_ppl=float("inf"),
            tok_per_sec=0.0,
            tokens_seen=0,
            n_batches=n_batches,
        )

    # Token-averaged loss (not batch-averaged — more stable)
    val_loss    = total_loss / total_active
    val_ppl     = math.exp(min(val_loss, 20.0))
    tok_per_sec = total_active / max(elapsed, 1e-6)

    model.train()

    logger.debug(
        "Validation — batches=%d active_tokens=%d total_tokens=%d",
        n_batches, total_active, total_tokens,
    )

    return ValidationResult(
        val_loss=val_loss,
        val_ppl=val_ppl,
        tok_per_sec=tok_per_sec,
        tokens_seen=total_active,
        n_batches=n_batches,
    )


# ---------------------------------------------------------------------------
# Model forward helper
# ---------------------------------------------------------------------------

def _forward(
    model:          nn.Module,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Call the model and return logits (B, T, V).

    The model is expected to accept ``(input_ids, attention_mask)`` or just
    ``(input_ids)``.  We try both so this works with the CyberSLM architecture
    as well as HuggingFace-style wrappers.
    """
    try:
        out = model(input_ids, attention_mask=attention_mask)
    except TypeError:
        out = model(input_ids)

    # Accept raw tensor or object with .logits attribute
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(
        f"Model output type {type(out)} not recognised. "
        "Expected a Tensor or an object with a .logits attribute."
    )
