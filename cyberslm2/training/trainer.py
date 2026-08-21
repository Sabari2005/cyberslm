"""
trainer.py
==========
Training loop for both pretraining and SFT.

Handles gradient accumulation, bf16 autocast, gradient clipping, the WSD
schedule, periodic evaluation, and resumable checkpointing.

Two batch shapes are supported:
  * pretraining -- a (B, seq_len+1) tensor of packed tokens, shifted internally
    and masked block-diagonally by document.
  * SFT -- a dict with input_ids / labels / attention_mask, masked causally plus
    key-padding.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from cyberslm2.configs.presets import ModelConfig, TrainConfig
from cyberslm2.data.packing import (
    build_doc_causal_mask,
    build_doc_ids,
    build_padding_causal_mask,
)
from cyberslm2.data.special_tokens import IGNORE_INDEX
from cyberslm2.training.loss import cross_entropy_with_z_loss, token_accuracy
from cyberslm2.training.optim import build_optimizer
from cyberslm2.training.schedule import apply_lr, build_schedule

log = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class Trainer:
    def __init__(
        self,
        model,
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
        mode: str = "pretrain",
    ) -> None:
        self.cfg = train_config
        self.mcfg = model_config
        self.mode = mode

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = build_optimizer(self.model, train_config)
        self.schedule = build_schedule(train_config)

        self.autocast_dtype = _DTYPES[train_config.dtype]
        # fp16 needs loss scaling to keep small gradients from flushing to zero;
        # bf16 has the same exponent range as fp32 and does not.
        self.use_scaler = (
            train_config.dtype == "float16" and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.step = 0
        self.data_epoch = 0
        self.best_val = float("inf")
        self._train_iter: Optional[Iterator] = None

        self.ckpt_dir = Path(train_config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- batch handling -----------------------------------------------------

    def _next_batch(self) -> Any:
        """Pull one batch, rolling to the next epoch when the loader runs dry."""
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            return next(self._train_iter)
        except StopIteration:
            self.data_epoch += 1
            ds = self.train_loader.dataset
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(self.data_epoch)
            log.info("Data epoch -> %d (reshuffled)", self.data_epoch)
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def _prepare(self, batch: Any) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """Return (input_ids, targets, attn_mask) already on-device and shifted."""
        if isinstance(batch, dict):
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            pad = batch.get("attention_mask")

            x, y = ids[:, :-1].contiguous(), labels[:, 1:].contiguous()
            mask = None
            if pad is not None:
                pad = pad.to(self.device, non_blocking=True)[:, :-1]
                mask = build_padding_causal_mask(pad)
            return x, y, mask

        tokens = batch.to(self.device, non_blocking=True)
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        mask = None
        if self.cfg.doc_aware_mask:
            mask = build_doc_causal_mask(build_doc_ids(x))
        return x, y, mask

    # -- training -----------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        self.model.train()

        log.info(
            "Training %s | %s | %d steps | %.2fM tokens/step | %.2fB tokens total",
            self.mode,
            self.device,
            cfg.max_steps,
            cfg.tokens_per_step() / 1e6,
            cfg.total_tokens() / 1e9,
        )

        t0 = time.perf_counter()
        running = 0.0

        while self.step < cfg.max_steps:
            lr_mult = apply_lr(self.optimizer, self.schedule, self.step)
            self.optimizer.zero_grad(set_to_none=True)

            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                batch = self._next_batch()
                x, y, mask = self._prepare(batch)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.device.type == "cuda",
                ):
                    logits, _ = self.model(x, attn_mask=mask)
                    loss, ce, _z = cross_entropy_with_z_loss(
                        logits, y, self.mcfg.z_loss_coef, IGNORE_INDEX
                    )

                # Scale so the accumulated gradient equals the mean over the
                # full effective batch rather than their sum.
                scaled = loss / cfg.grad_accum_steps
                if self.use_scaler:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                accum_loss += ce.item() / cfg.grad_accum_steps

            if self.use_scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )

            if self.use_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.step += 1
            running += accum_loss

            if self.step % cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                tps = cfg.tokens_per_step() * cfg.log_every / max(elapsed, 1e-9)
                mean_loss = running / cfg.log_every
                log.info(
                    "step %6d/%d | loss %.4f | ppl %8.2f | lr x%.3f | "
                    "gnorm %.2f | %.0f tok/s",
                    self.step, cfg.max_steps, mean_loss,
                    math.exp(min(mean_loss, 20)), lr_mult,
                    float(grad_norm), tps,
                )
                running = 0.0
                t0 = time.perf_counter()

            if self.val_loader is not None and self.step % cfg.eval_every == 0:
                val = self.evaluate()
                log.info(
                    "step %6d | VAL loss %.4f | ppl %.2f | acc %.4f",
                    self.step, val["loss"], val["ppl"], val["accuracy"],
                )
                if val["loss"] < self.best_val:
                    self.best_val = val["loss"]
                    self.save("best", val)
                self.model.train()
                t0 = time.perf_counter()

            if self.step % cfg.save_every == 0:
                self.save("latest")
                t0 = time.perf_counter()

        self.save("final")
        log.info("Training complete at step %d (best val %.4f)", self.step, self.best_val)

    # -- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """
        Token-weighted validation loss.

        Averaging per-batch means would weight a batch with few unmasked targets
        the same as a full one, which skews the number on SFT data where the
        trainable fraction varies a lot between examples.
        """
        if self.val_loader is None:
            return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0}

        self.model.eval()
        total_nll, total_tokens, total_correct = 0.0, 0, 0

        for i, batch in enumerate(self.val_loader):
            if 0 < self.cfg.eval_batches <= i:
                break
            x, y, mask = self._prepare(batch)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.device.type == "cuda",
            ):
                logits, _ = self.model(x, attn_mask=mask)

            n = int((y != IGNORE_INDEX).sum().item())
            if n == 0:
                continue
            _, ce, _ = cross_entropy_with_z_loss(logits, y, 0.0, IGNORE_INDEX)
            total_nll += ce.item() * n
            total_tokens += n
            total_correct += int(
                token_accuracy(logits, y, IGNORE_INDEX) * n
            )

        if total_tokens == 0:
            return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0}

        loss = total_nll / total_tokens
        return {
            "loss": loss,
            "ppl": math.exp(min(loss, 20)),
            "accuracy": total_correct / total_tokens,
        }

    # -- checkpointing ------------------------------------------------------

    def save(self, tag: str, metrics: Optional[dict] = None) -> Path:
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "data_epoch": self.data_epoch,
                "best_val": self.best_val,
                "model_config": self.mcfg.to_dict(),
                "train_config": self.cfg.to_dict(),
                "metrics": metrics or {},
            },
            path,
        )
        (self.ckpt_dir / f"{tag}_config.json").write_text(
            json.dumps(
                {"model": self.mcfg.to_dict(), "train": self.cfg.to_dict(),
                 "step": self.step, "metrics": metrics or {}},
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("Saved checkpoint -> %s", path)
        return path

    def load(self, path: str | Path, resume_optimizer: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if resume_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt.get("step", 0)
        self.data_epoch = ckpt.get("data_epoch", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        log.info("Resumed from %s at step %d", path, self.step)
