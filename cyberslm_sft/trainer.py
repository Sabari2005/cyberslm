"""
CyberSLM SFT — Trainer
=======================
Orchestrates the full supervised instruction fine-tuning loop.

Training flow
-------------
1. Load pretrained checkpoint into model.
2. Build DataLoaders from pre-tokenised SFTDatasets.
3. Build AdamW optimizer + cosine LR scheduler.
4. For each epoch:
   a. Iterate batches with gradient accumulation.
   b. Clip gradients; step optimizer and scheduler.
   c. Log every ``cfg.train.log_every_n_steps`` steps.
   d. Validate every ``cfg.train.eval_every_n_steps`` steps (or at epoch end).
   e. Save latest checkpoint; save best checkpoint on val_loss improvement.
5. Run inference sanity checks after training.

Resume
------
If ``cfg.train.resume_from_checkpoint`` is set, the trainer loads the
optimizer state, scheduler state, epoch, and global step from the checkpoint
before the first epoch so training continues seamlessly.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.sft_config import SFTConfig
from data.collator import SFTCollator
from data.loss_masking import masked_cross_entropy, count_response_tokens, LossMaskAuditor
from data.prompt_formatter import IGNORE_INDEX
from data.sft_dataset import SFTDataset
from utils.logging_utils import TrainingLogger, _safe_perplexity
from utils.optimizer import build_optimizer, build_scheduler, clip_gradients
from utils.validation import ValidationResult, run_validation, _forward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training state (survives checkpoint round-trips)
# ---------------------------------------------------------------------------

class TrainingState:
    """Mutable state that is serialised into every checkpoint."""

    def __init__(self) -> None:
        self.epoch:          int   = 0
        self.global_step:    int   = 0
        self.best_val_loss:  float = float("inf")
        self.train_start:    float = time.time()

    def state_dict(self) -> dict:
        return {
            "epoch":         self.epoch,
            "global_step":   self.global_step,
            "best_val_loss": self.best_val_loss,
        }

    def load_state_dict(self, d: dict) -> None:
        self.epoch        = d.get("epoch",         0)
        self.global_step  = d.get("global_step",   0)
        self.best_val_loss= d.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Checkpoint helpers (inline — CheckpointManager is in Phase 4)
# ---------------------------------------------------------------------------

def _save_checkpoint(
    output_dir:   str | Path,
    tag:          str,
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scheduler:    torch.optim.lr_scheduler.LRScheduler,
    state:        TrainingState,
    cfg:          SFTConfig,
) -> Path:
    """
    Save a checkpoint to ``<output_dir>/<tag>/``.

    Saves:
    * ``model.pt``      — model state dict
    * ``optimizer.pt``  — optimizer state dict
    * ``scheduler.pt``  — scheduler state dict
    * ``training_state.pt`` — epoch, step, best_val_loss
    * ``sft_config.json``   — full config (for reproducibility)

    Returns the checkpoint directory path.
    """
    from configs.sft_config import save_config

    ckpt_dir = Path(output_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(),     ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    torch.save(state.state_dict(),     ckpt_dir / "training_state.pt")
    save_config(cfg, ckpt_dir / "sft_config.json")

    logger.info("Checkpoint saved → %s", ckpt_dir)
    return ckpt_dir


def _load_checkpoint(
    ckpt_dir:  str | Path,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state:     TrainingState,
    device:    torch.device,
) -> None:
    """
    Load optimizer, scheduler, and training state from a checkpoint directory.
    The model weights are assumed to be already loaded (from pretrained ckpt).
    """
    ckpt_dir = Path(ckpt_dir)

    model_path = ckpt_dir / "model.pt"
    if model_path.exists():
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        logger.info("Model weights loaded from checkpoint: %s", model_path)

    opt_path = ckpt_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        logger.info("Optimizer state restored from: %s", opt_path)

    sched_path = ckpt_dir / "scheduler.pt"
    if sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location=device))
        logger.info("Scheduler state restored from: %s", sched_path)

    ts_path = ckpt_dir / "training_state.pt"
    if ts_path.exists():
        state.load_state_dict(torch.load(ts_path, map_location="cpu"))
        logger.info(
            "Resumed from epoch=%d step=%d best_val_loss=%.4f",
            state.epoch, state.global_step, state.best_val_loss,
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SFTTrainer:
    """
    Supervised Instruction Fine-Tuning trainer for CyberSLM.

    Parameters
    ----------
    model:       The pretrained CyberSLM model (weights already loaded).
    train_ds:    Pre-tokenised training ``SFTDataset``.
    val_ds:      Pre-tokenised validation ``SFTDataset``.
    cfg:         Master ``SFTConfig``.
    device:      ``torch.device`` to train on.
    tokenizer:   Tokenizer instance (used only for inference tests).
    """

    def __init__(
        self,
        model:     nn.Module,
        train_ds:  SFTDataset,
        val_ds:    SFTDataset,
        cfg:       SFTConfig,
        device:    torch.device,
        tokenizer,
    ) -> None:
        self.model     = model.to(device)
        self.train_ds  = train_ds
        self.val_ds    = val_ds
        self.cfg       = cfg
        self.device    = device
        self.tokenizer = tokenizer
        self.state     = TrainingState()

        # --- DataLoaders ---
        collator = SFTCollator(
            pad_id=tokenizer.pad_id,
            max_seq_len=cfg.data.max_seq_len,
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.per_device_batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            collate_fn=collator,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.per_device_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collator,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
            drop_last=False,
        )

        # --- Steps / schedule ---
        steps_per_epoch      = math.ceil(
            len(train_ds) / cfg.train.per_device_batch_size
        )
        self.steps_per_epoch = steps_per_epoch
        self.total_steps     = math.ceil(
            steps_per_epoch / cfg.train.gradient_accumulation_steps
        ) * cfg.train.num_epochs

        # --- Mixed precision ---
        # cfg.train.dtype used to be logged and then ignored entirely, so SFT
        # always ran in fp32 no matter what the config said.
        from cyberslm.training.trainer import resolve_amp
        self.amp_dtype, _needs_scaler = resolve_amp(cfg.train.dtype, device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=_needs_scaler)
        logger.info(
            "Mixed precision: %s",
            "disabled (float32)" if self.amp_dtype is None
            else str(self.amp_dtype).replace("torch.", ""),
        )

        # --- Optimizer & scheduler ---
        self.optimizer = build_optimizer(model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train, self.total_steps)

        # --- Logger ---
        warmup_steps = max(1, int(cfg.train.warmup_ratio * self.total_steps))
        self.tlogger  = TrainingLogger(
            total_steps=self.total_steps,
            log_every_n=cfg.train.log_every_n_steps,
        )
        self.tlogger.log_training_start(
            run_name=cfg.train.run_name,
            n_train=len(train_ds),
            n_val=len(val_ds),
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
            device=str(device),
            dtype=cfg.train.dtype,
        )

        # --- Auditor (masking quality) ---
        self.mask_auditor = LossMaskAuditor()

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop and save the final checkpoint."""
        cfg   = self.cfg.train
        state = self.state

        run_start = time.time()

        # ---- Resume ----
        if cfg.resume_from_checkpoint:
            _load_checkpoint(
                cfg.resume_from_checkpoint,
                self.model, self.optimizer, self.scheduler, state,
                self.device,
            )

        self.model.train()

        for epoch in range(state.epoch, cfg.num_epochs):
            state.epoch = epoch
            epoch_start = time.time()
            epoch_loss  = 0.0
            epoch_windows = 0

            self.tlogger.reset_throughput_timer()

            # Accumulation is driven by explicit WINDOWS of micro-batches rather
            # than by `batch_idx % accum == 0`. Two bugs came from the old form:
            #   * a `continue` on a skipped batch could land on the accumulation
            #     boundary, silently dropping that optimizer step and carrying
            #     partial gradients into the next window;
            #   * the reported loss was whatever the LAST micro-batch happened to
            #     be, not the mean over the window.
            micro_iter = iter(self.train_loader)
            n_micro    = len(self.train_loader)
            consumed   = 0

            while consumed < n_micro:
                window = []
                for _ in range(cfg.gradient_accumulation_steps):
                    if consumed >= n_micro:
                        break
                    window.append(next(micro_iter))
                    consumed += 1
                if not window:
                    break

                # Count supervised targets AFTER the causal shift, because that
                # is exactly the denominator masked_cross_entropy uses.
                tok_counts = [
                    int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in window
                ]
                total_tok = sum(tok_counts)
                for b in window:
                    self.mask_auditor.update(b["labels"])

                if total_tok == 0:
                    logger.warning(
                        "Window at step %d has no supervised tokens - skipped.",
                        state.global_step,
                    )
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                window_loss_sum = 0.0

                for batch, n_tok in zip(window, tok_counts):
                    if n_tok == 0:
                        continue
                    input_ids      = batch["input_ids"].to(self.device)
                    labels         = batch["labels"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.amp_dtype is not None,
                    ):
                        logits = _forward(self.model, input_ids, attention_mask)
                        loss_sum = masked_cross_entropy(logits, labels, reduction="sum")

                    # Dividing the SUM by the window's total token count weights
                    # every supervised token equally. Averaging per micro-batch
                    # and dividing by accum_steps (the old behaviour) gave a
                    # sample with 5 response tokens the same pull on the gradient
                    # as one with 500.
                    self.scaler.scale(loss_sum / total_tok).backward()
                    window_loss_sum += loss_sum.item()

                train_loss = window_loss_sum / total_tok

                self.scaler.unscale_(self.optimizer)
                grad_norm = clip_gradients(self.model, cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                state.global_step += 1
                epoch_loss    += train_loss
                epoch_windows += 1
                self.tlogger.accumulate_tokens(total_tok)

                self.tlogger.log_step(
                    step=state.global_step,
                    loss=train_loss,
                    lr=self.scheduler.get_last_lr()[0],
                    grad_norm=grad_norm,
                    elapsed_total=time.time() - run_start,
                )

                if (
                    cfg.eval_every_n_steps > 0
                    and state.global_step % cfg.eval_every_n_steps == 0
                ):
                    self._run_and_log_validation(state, run_start)

                if (
                    cfg.save_every_n_steps > 0
                    and state.global_step % cfg.save_every_n_steps == 0
                ):
                    _save_checkpoint(
                        cfg.output_dir, "latest",
                        self.model, self.optimizer, self.scheduler, state, self.cfg,
                    )

            # ---- End of epoch ----
            avg_epoch_loss = epoch_loss / max(epoch_windows, 1)
            self.tlogger.log_epoch_end(
                epoch=epoch + 1,
                train_loss=avg_epoch_loss,
                elapsed=time.time() - epoch_start,
            )

            # Validate at epoch end (if not already done mid-epoch)
            self._run_and_log_validation(state, run_start)

            # Always save latest at epoch end
            _save_checkpoint(
                cfg.output_dir, "latest",
                self.model, self.optimizer, self.scheduler, state, self.cfg,
            )

        # ---- Training complete ----
        self.mask_auditor.log_summary()
        self.tlogger.log_training_end(
            best_val_loss=state.best_val_loss,
            elapsed=time.time() - run_start,
        )

        # ---- Inference sanity check ----
        if cfg.run_inference_test:
            from utils.inference import run_inference_tests
            run_inference_tests(
                model=self.model,
                tokenizer=self.tokenizer,
                cfg=self.cfg,
                device=self.device,
            )

    # ------------------------------------------------------------------
    def _run_and_log_validation(
        self,
        state:     TrainingState,
        run_start: float,
    ) -> ValidationResult:
        result = run_validation(
            model=self.model,
            val_loader=self.val_loader,
            device=self.device,
            amp_dtype=self.amp_dtype,
        )

        is_best = result.val_loss < state.best_val_loss
        if is_best:
            state.best_val_loss = result.val_loss
            _save_checkpoint(
                self.cfg.train.output_dir, "best",
                self.model, self.optimizer, self.scheduler, state, self.cfg,
            )
            logger.info(
                "New best checkpoint saved (val_loss=%.4f, ppl=%.2f)",
                result.val_loss, result.val_ppl,
            )

        self.tlogger.log_validation(
            step=state.global_step,
            epoch=state.epoch + 1,
            val_loss=result.val_loss,
            val_ppl=result.val_ppl,
            tok_per_sec=result.tok_per_sec,
            is_best=is_best,
        )

        self.model.train()
        return result
