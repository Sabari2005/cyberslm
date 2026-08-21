"""
CyberSLM SFT — Logging
=======================
Configures Python's standard ``logging`` module and provides structured
helper functions that emit consistent, machine-parseable log lines.

All training metrics are emitted as a single line with labelled fields so
they can be grepped, piped to a CSV, or fed into a log monitor without
post-processing.

Example log line::

    [STEP  150/3000] loss=1.4321 ppl=4.19 lr=1.87e-05 gnorm=0.423 tok/s=12450 elapsed=45.2s

"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Root logger setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_file:  Optional[str] = None,
    log_level: int = logging.INFO,
) -> None:
    """
    Configure the root logger with a console handler and an optional file
    handler.  Call once at the start of the training script.

    Parameters
    ----------
    log_file:  If provided, also write logs to this path.
    log_level: Verbosity level (``logging.DEBUG``, ``logging.INFO``, …).
    """
    fmt     = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear any pre-existing handlers to avoid duplicate output
    root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        logging.getLogger(__name__).info("Logging to file: %s", log_file)


# ---------------------------------------------------------------------------
# Structured metric logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Emits structured log lines for training and validation metrics.

    Parameters
    ----------
    total_steps:    Total gradient update steps in the run (for progress).
    log_every_n:    Emit a training log every N optimiser steps.
    """

    def __init__(self, total_steps: int, log_every_n: int = 10) -> None:
        self.total_steps = total_steps
        self.log_every_n = log_every_n
        self._step_start: float = time.time()
        self._tokens_since_log: int = 0

    # ------------------------------------------------------------------
    def reset_throughput_timer(self) -> None:
        """Call at the start of each logging window."""
        self._step_start       = time.time()
        self._tokens_since_log = 0

    def accumulate_tokens(self, n_tokens: int) -> None:
        """Accumulate token count for throughput calculation."""
        self._tokens_since_log += n_tokens

    # ------------------------------------------------------------------
    def log_step(
        self,
        step:       int,
        loss:       float,
        lr:         float,
        grad_norm:  float,
        elapsed_total: float,
    ) -> None:
        """
        Emit a training-step log line.

        Parameters
        ----------
        step:          Current optimiser step (1-indexed).
        loss:          Batch loss value.
        lr:            Current learning rate.
        grad_norm:     Pre-clip gradient norm.
        elapsed_total: Seconds elapsed since the run started.
        """
        if step % self.log_every_n != 0:
            return

        elapsed_window = time.time() - self._step_start
        tok_per_sec    = self._tokens_since_log / max(elapsed_window, 1e-6)
        ppl            = _safe_perplexity(loss)

        logger.info(
            "[STEP %5d/%d] loss=%.4f ppl=%.2f lr=%.3e "
            "gnorm=%.3f tok/s=%d elapsed=%.1fs",
            step, self.total_steps,
            loss, ppl, lr,
            grad_norm, int(tok_per_sec), elapsed_total,
        )
        self.reset_throughput_timer()

    # ------------------------------------------------------------------
    def log_validation(
        self,
        step:       int,
        epoch:      int,
        val_loss:   float,
        val_ppl:    float,
        tok_per_sec: float,
        is_best:    bool,
    ) -> None:
        """Emit a validation summary line."""
        best_tag = " *** BEST ***" if is_best else ""
        logger.info(
            "[VAL   epoch=%d step=%d] loss=%.4f ppl=%.2f tok/s=%d%s",
            epoch, step, val_loss, val_ppl, int(tok_per_sec), best_tag,
        )

    # ------------------------------------------------------------------
    def log_epoch_end(
        self,
        epoch:      int,
        train_loss: float,
        elapsed:    float,
    ) -> None:
        logger.info(
            "[EPOCH %d END] avg_train_loss=%.4f elapsed=%.1fs",
            epoch, train_loss, elapsed,
        )

    # ------------------------------------------------------------------
    def log_training_start(
        self,
        run_name:      str,
        n_train:       int,
        n_val:         int,
        total_steps:   int,
        warmup_steps:  int,
        device:        str,
        dtype:         str,
    ) -> None:
        logger.info("=" * 60)
        logger.info("CyberSLM Instruction Fine-Tuning")
        logger.info("Run name    : %s", run_name)
        logger.info("Train size  : %d samples", n_train)
        logger.info("Val size    : %d samples", n_val)
        logger.info("Total steps : %d", total_steps)
        logger.info("Warmup steps: %d", warmup_steps)
        logger.info("Device      : %s", device)
        logger.info("Dtype       : %s", dtype)
        logger.info("=" * 60)

    def log_training_end(self, best_val_loss: float, elapsed: float) -> None:
        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info("Best validation loss: %.4f  (ppl=%.2f)",
                    best_val_loss, _safe_perplexity(best_val_loss))
        logger.info("Total time: %.1f s (%.1f min)", elapsed, elapsed / 60.0)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_perplexity(loss: float) -> float:
    """Return e^loss clamped to avoid overflow for very high losses."""
    import math
    try:
        return math.exp(min(loss, 20.0))
    except OverflowError:
        return float("inf")
