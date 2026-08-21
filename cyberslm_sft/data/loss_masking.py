"""
CyberSLM SFT — Loss Masking
============================
Utilities for computing and auditing the cross-entropy loss with
per-token masking.

Design
------
``nn.CrossEntropyLoss(ignore_index=-100)`` already skips ``-100`` labels
natively.  This module provides:

1. ``masked_cross_entropy`` — a thin wrapper that enforces the correct
   ignore index, logs a warning when an entire batch is masked (which
   would produce a NaN loss and silently stall training), and returns a
   scalar loss suitable for ``.backward()``.

2. ``count_response_tokens`` — diagnostic helper that counts how many
   tokens per batch are actually trained on (label ≠ -100).

3. ``verify_masking`` — sanity-check that inspects a single
   ``(input_ids, labels)`` pair and returns a human-readable report.
   Called once per run at startup.

4. ``LossMaskAuditor`` — accumulates masking statistics across many
   batches and emits a summary, helping catch dataset quality issues
   (e.g. too many fully-masked samples).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.prompt_formatter import IGNORE_INDEX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core loss function
# ---------------------------------------------------------------------------

def masked_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Cross-entropy loss with ``ignore_index=-100``.

    Parameters
    ----------
    logits:    ``(B, T, V)``  — raw model output before softmax.
    labels:    ``(B, T)``     — target token ids; -100 positions are ignored.
    reduction: ``"mean"`` averages over all *non-masked* positions.
               ``"sum"``  sums them.  ``"none"`` returns per-token losses.

    Returns
    -------
    Scalar loss tensor (``reduction="mean"`` or ``"sum"``) or
    ``(B, T)`` tensor (``reduction="none"``).

    Raises
    ------
    RuntimeError
        If every label in the batch is ``IGNORE_INDEX`` (nothing to train on).
        This prevents a silent NaN loss.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be 3-D (B, T, V), got shape {tuple(logits.shape)}"
        )
    if labels.dim() != 2:
        raise ValueError(
            f"labels must be 2-D (B, T), got shape {tuple(labels.shape)}"
        )

    B, T, V = logits.shape

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten for F.cross_entropy
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)

    # Count non-masked tokens for the guard
    n_active = (flat_labels != IGNORE_INDEX).sum().item()
    if n_active == 0:
        raise RuntimeError(
            "All labels in this batch are IGNORE_INDEX (-100). "
            "The loss would be undefined (NaN). "
            "Check your prompt formatter and dataset."
        )

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )

    if reduction == "none":
        # After the causal shift the sequence length is T-1, not T.
        loss = loss.reshape(B, T - 1)

    return loss


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def count_response_tokens(labels: Tensor) -> dict:
    """
    Return a breakdown of label statistics for a batch.

    Parameters
    ----------
    labels: ``(B, T)`` label tensor.

    Returns
    -------
    dict with keys:
        ``total``         — B × T
        ``active``        — tokens where label ≠ -100 (trained on)
        ``masked``        — tokens where label == -100 (ignored)
        ``active_ratio``  — fraction of tokens contributing to loss
    """
    total   = labels.numel()
    active  = int((labels != IGNORE_INDEX).sum().item())
    masked  = total - active

    return {
        "total":        total,
        "active":       active,
        "masked":       masked,
        "active_ratio": active / max(total, 1),
    }


@dataclass
class MaskReport:
    """Human-readable masking report for a single sample."""
    n_tokens:      int
    n_active:      int
    n_masked:      int
    active_ratio:  float
    prompt_end_idx: int          # first token that is NOT masked
    sample_type:   str           # "alpaca" or "conversation"


def verify_masking(
    input_ids: List[int],
    labels:    List[int],
    tokenizer,
    sample_type: str = "alpaca",
) -> MaskReport:
    """
    Inspect a single ``(input_ids, labels)`` pair and return a report.

    The function also emits ``DEBUG`` log lines showing which decoded
    tokens are masked vs. active — useful for a one-shot startup sanity
    check.

    Parameters
    ----------
    input_ids:   Raw token ids (before padding).
    labels:      Corresponding labels (-100 for masked positions).
    tokenizer:   Any object with a ``.decode(ids) -> str`` method.
    sample_type: ``"alpaca"`` or ``"conversation"`` (for the report).
    """
    assert len(input_ids) == len(labels), (
        "input_ids and labels must have the same length"
    )

    n_tokens = len(input_ids)
    n_active = sum(1 for l in labels if l != IGNORE_INDEX)
    n_masked = n_tokens - n_active

    # Find first active position (i.e. where prompt ends)
    prompt_end = n_tokens  # default: fully masked
    for i, l in enumerate(labels):
        if l != IGNORE_INDEX:
            prompt_end = i
            break

    active_ratio = n_active / max(n_tokens, 1)

    logger.debug("=== Loss-mask verification (%s) ===", sample_type)
    logger.debug("Total tokens : %d", n_tokens)
    logger.debug("Active tokens: %d (%.1f %%)", n_active, 100 * active_ratio)
    logger.debug("Masked tokens: %d", n_masked)

    # Decode and display the boundary region (5 tokens either side)
    lo = max(0, prompt_end - 5)
    hi = min(n_tokens, prompt_end + 5)
    boundary_ids    = input_ids[lo:hi]
    boundary_labels = labels[lo:hi]

    for offset, (tid, lbl) in enumerate(zip(boundary_ids, boundary_labels)):
        tok   = tokenizer.decode([tid])
        state = "ACTIVE" if lbl != IGNORE_INDEX else "masked"
        logger.debug("  [%3d] %-10s %r", lo + offset, state, tok)

    return MaskReport(
        n_tokens=n_tokens,
        n_active=n_active,
        n_masked=n_masked,
        active_ratio=active_ratio,
        prompt_end_idx=prompt_end,
        sample_type=sample_type,
    )


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

@dataclass
class LossMaskAuditor:
    """
    Accumulates masking statistics across batches and surfaces anomalies.

    Usage::

        auditor = LossMaskAuditor()
        for batch in loader:
            auditor.update(batch["labels"])
        auditor.log_summary()
    """
    _total_tokens:  int = field(default=0, init=False)
    _active_tokens: int = field(default=0, init=False)
    _n_batches:     int = field(default=0, init=False)
    _fully_masked:  int = field(default=0, init=False)  # batches with 0 active

    def update(self, labels: Tensor) -> None:
        stats = count_response_tokens(labels)
        self._total_tokens  += stats["total"]
        self._active_tokens += stats["active"]
        self._n_batches     += 1
        if stats["active"] == 0:
            self._fully_masked += 1

    def log_summary(self) -> None:
        active_ratio = self._active_tokens / max(self._total_tokens, 1)
        logger.info(
            "LossMaskAuditor — batches: %d | tokens: %d | "
            "active: %d (%.1f %%) | fully-masked batches: %d",
            self._n_batches,
            self._total_tokens,
            self._active_tokens,
            100.0 * active_ratio,
            self._fully_masked,
        )
        if self._fully_masked > 0:
            logger.warning(
                "%d batch(es) had zero active tokens and were skipped. "
                "Check that your sequences are not being truncated before "
                "the response begins.",
                self._fully_masked,
            )
        if active_ratio < 0.05:
            logger.warning(
                "Only %.1f %% of tokens contribute to the loss. "
                "Consider shortening prompts or increasing max_seq_len.",
                100.0 * active_ratio,
            )
