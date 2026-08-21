"""
CyberSLM Training Engine
=========================
Production-quality pretraining loop for the CyberSLM decoder-only transformer.

Features
--------
- Cross-entropy language model loss
- Gradient accumulation (configurable micro-batches per optimizer step)
- Gradient clipping (global norm)
- Linear warmup + cosine decay learning-rate schedule
- Periodic validation with best-checkpoint tracking
- Full checkpoint save / resume (model + optimizer + scheduler + RNG state)
- Deterministic training with seed control
- Single-GPU default; DDP opt-in via TrainingConfig.distributed
- torch.compile opt-in via TrainingConfig.compile_model
- Structured per-step metrics (loss, grad norm, tokens/s, ETA, GPU/CPU mem)

Integration contract
--------------------
The caller is responsible for providing a DataLoader whose batches are
``torch.Tensor`` of shape ``(batch_size, seq_len)`` with dtype ``torch.long``.
The training engine calls ``next(train_iter)`` to get micro-batches; when the
iterator is exhausted it is recreated from the DataLoader.

Distributed usage
-----------------
Launch with torchrun (not torch.multiprocessing.spawn):

    torchrun --nproc_per_node=NUM_GPUS train.py

The engine reads LOCAL_RANK, RANK, WORLD_SIZE from the environment.
DDP is initialised inside :meth:`Trainer.train` when
``train_config.distributed`` is True.  All checkpoint I/O and logging is
gated on ``rank == 0``.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from cyberslm.model.config import CyberSLMConfig
from cyberslm.model.model import CyberSLM, build_model, count_parameters
from cyberslm.training.config import TrainingConfig
from cyberslm.training.scheduler import CosineWarmupScheduler
from cyberslm.training.checkpoint import CheckpointManager
from cyberslm.training.metrics import StepMetrics, get_gpu_mem_gb, get_cpu_mem_gb

log = logging.getLogger(__name__)


_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_amp(dtype_name: str, device: torch.device):
    """
    Map a config dtype string to (autocast_dtype, needs_grad_scaler).

    bfloat16 has the same exponent range as float32, so it needs no loss
    scaling. float16 does. On CPU (or when float32 is requested) autocast is
    disabled entirely and this returns (None, False).
    """
    name = (dtype_name or "float32").lower()
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype {dtype_name!r}; choose from {sorted(_DTYPES)}")
    if name == "float32" or device.type != "cuda":
        return None, False
    if name == "bfloat16" and not torch.cuda.is_bf16_supported():
        log.warning("bfloat16 unsupported on this GPU -- falling back to float16.")
        return torch.float16, True
    return _DTYPES[name], name == "float16"


# --------------------------------------------------------------------------- #
# Seed                                                                          #
# --------------------------------------------------------------------------- #

def set_seed(seed: int, rank: int = 0) -> None:
    """
    Set all RNG seeds deterministically.

    Each DDP rank gets a unique seed derived from the base seed so that
    different ranks draw different data batches.

    Parameters
    ----------
    seed : int
        Base seed.
    rank : int
        DDP rank (0 for single-GPU or CPU training).
    """
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


# --------------------------------------------------------------------------- #
# Loss                                                                          #
# --------------------------------------------------------------------------- #

def compute_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute next-token prediction cross-entropy loss.

    Parameters
    ----------
    logits : Tensor
        Shape ``(batch, seq_len, vocab_size)`` — raw model outputs.
    targets : Tensor
        Shape ``(batch, seq_len)`` — target token IDs.

    Returns
    -------
    Tensor
        Scalar mean cross-entropy loss.
    """
    B, T, V = logits.shape
    # ignore_index guards against ever training on <pad> (id 0 is a real vocab
    # row). Packed pretraining batches contain no padding today, so this is a
    # no-op now and a safety net the moment a padded batch appears.
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="mean",
        ignore_index=ignore_index,
    )
    return loss


# --------------------------------------------------------------------------- #
# Optimizer builder                                                             #
# --------------------------------------------------------------------------- #

def build_optimizer(
    model: nn.Module,
    train_config: TrainingConfig,
) -> optim.AdamW:
    """
    Build AdamW optimizer with weight-decay separation.

    Weight decay is applied only to weight matrices (2-D tensors).
    Bias terms, RMSNorm scale weights (1-D), and embedding vectors
    are excluded from weight decay to avoid penalising learned scales.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters will be optimised.
    train_config : TrainingConfig

    Returns
    -------
    optim.AdamW
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay on 1-D tensors (biases, RMSNorm scales) or on the
        # token embedding. The embedding is weight-tied to the LM head, so
        # decaying it also shrinks the output projection / logit scale; the
        # common convention is to leave (un)embeddings undecayed.
        is_embedding = ("embedding" in name) or name.endswith("lm_head.weight")
        if param.ndim >= 2 and not is_embedding:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {
            "params":       decay_params,
            "weight_decay": train_config.weight_decay,
        },
        {
            "params":       no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        param_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.eps,
        fused=torch.cuda.is_available(),  # fused kernel for speed on CUDA
    )
    return optimizer


# --------------------------------------------------------------------------- #
# Trainer                                                                       #
# --------------------------------------------------------------------------- #

class Trainer:
    """
    Manages the full pretraining loop for CyberSLM.

    Parameters
    ----------
    model_config : CyberSLMConfig
        Validated model configuration.
    train_config : TrainingConfig
        Validated training configuration.
    train_loader : DataLoader
        DataLoader for training data.  Batches: ``(batch_size, seq_len)`` long.
    val_loader : DataLoader
        DataLoader for validation data.
    resume_from : str | Path, optional
        Path to a checkpoint to resume from.  If None, starts from scratch.
    """

    def __init__(
        self,
        model_config: CyberSLMConfig,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str | Path] = None,
    ) -> None:
        train_config.validate()

        self.model_config  = model_config
        self.train_config  = train_config
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.resume_from   = Path(resume_from) if resume_from else None

        # ------------------------------------------------------------------ #
        # Distributed setup                                                    #
        # ------------------------------------------------------------------ #
        self.distributed = train_config.distributed
        if self.distributed:
            dist.init_process_group(backend="nccl")
            self.rank       = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
        else:
            self.rank       = 0
            self.local_rank = 0
            self.world_size = 1

        self.is_main = (self.rank == 0)

        # ------------------------------------------------------------------ #
        # Device                                                               #
        # ------------------------------------------------------------------ #
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # ------------------------------------------------------------------ #
        # Seed                                                                 #
        # ------------------------------------------------------------------ #
        set_seed(train_config.seed, rank=self.rank)

        # ------------------------------------------------------------------ #
        # Model                                                                #
        # ------------------------------------------------------------------ #
        self.model = build_model(model_config, device=self.device)

        if train_config.compile_model:
            log.info("Compiling model with torch.compile …")
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # ------------------------------------------------------------------ #
        # Mixed precision                                                      #
        # ------------------------------------------------------------------ #
        self.amp_dtype, needs_scaler = resolve_amp(train_config.dtype, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
        if self.amp_dtype is not None:
            log.info("Mixed precision: %s (grad_scaler=%s)",
                     str(self.amp_dtype).replace("torch.", ""), needs_scaler)
        else:
            log.info("Mixed precision: disabled (float32)")

        # ------------------------------------------------------------------ #
        # Optimizer & Scheduler                                                #
        # ------------------------------------------------------------------ #
        self.optimizer = build_optimizer(self.model, train_config)
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            peak_lr=train_config.learning_rate,
            min_lr=train_config.min_lr,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
        )

        # ------------------------------------------------------------------ #
        # Checkpoint manager                                                   #
        # ------------------------------------------------------------------ #
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=train_config.checkpoint_dir,
            keep_last_n=train_config.keep_last_n,
            save_best=train_config.save_best,
        )

        # ------------------------------------------------------------------ #
        # State                                                                #
        # ------------------------------------------------------------------ #
        self.global_step    = 0
        self.tokens_seen    = 0
        self.best_val_loss  = float("inf")
        self._data_epoch    = 0
        self._train_iter: Optional[Iterator] = None

        # ------------------------------------------------------------------ #
        # Resume                                                               #
        # ------------------------------------------------------------------ #
        if self.resume_from is not None:
            self._resume(self.resume_from)

        if self.is_main:
            param_info = count_parameters(
                self.model.module if hasattr(self.model, "module") else self.model
            )
            log.info(
                f"CyberSLM ready  |  params={param_info['total']:,}  |  "
                f"device={self.device}  |  distributed={self.distributed}"
            )
            log.info(str(train_config))

    # ---------------------------------------------------------------------- #
    # Public: train                                                            #
    # ---------------------------------------------------------------------- #

    def train(self) -> None:
        """
        Run the full training loop from ``global_step`` to ``max_steps``.

        The loop structure is:
          for each optimizer step:
            for each micro-batch (gradient accumulation):
              forward → loss / grad_accum_steps → backward
            clip gradients → optimizer step → scheduler step
            log / validate / checkpoint on schedule
        """
        cfg          = self.train_config
        model        = self.model
        optimizer    = self.optimizer
        scheduler    = self.scheduler
        accum_steps  = cfg.grad_accum_steps

        model.train()
        self._train_iter = self._new_train_iter()
        t0           = time.perf_counter()
        step_t0      = t0
        running_loss = 0.0

        log.info(
            f"Training  |  start_step={self.global_step}  "
            f"max_steps={cfg.max_steps}"
        )

        while self.global_step < cfg.max_steps:

            optimizer.zero_grad(set_to_none=True)

            # ---- Gradient accumulation ------------------------------------ #
            accum_loss = 0.0
            for micro_step in range(accum_steps):
                x, y = self._next_batch(self._train_iter)
                x = x.to(self.device)
                y = y.to(self.device)

                # In DDP, only sync gradients on the last micro-step.
                sync_ctx = (
                    model.no_sync()
                    if self.distributed and micro_step < accum_steps - 1
                    else _null_context()
                )

                with sync_ctx:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.amp_dtype is not None,
                    ):
                        logits, _ = model(x)
                        loss = compute_lm_loss(logits, y)
                    # Divide before backward so gradients are already averaged.
                    self.scaler.scale(loss / accum_steps).backward()

                accum_loss += loss.item() / accum_steps

            # ---- Gradient clipping ---------------------------------------- #
            # Unscale first, otherwise the norm is measured on scaled grads and
            # the clip threshold means nothing. No-op when the scaler is off.
            self.scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            ).item()

            # ---- Optimizer step ------------------------------------------- #
            # Set the LR for THIS update before stepping, so update #1 uses
            # get_lr(1) (start of warmup) rather than get_lr(0) == 0.
            lr = scheduler.step()
            self.scaler.step(optimizer)
            self.scaler.update()

            self.global_step += 1
            tokens_this_step  = cfg.batch_size * cfg.seq_len * accum_steps * self.world_size
            self.tokens_seen += tokens_this_step
            running_loss      = accum_loss

            # ---- Logging -------------------------------------------------- #
            if self.is_main and self.global_step % cfg.log_every_steps == 0:
                now      = time.perf_counter()
                elapsed  = now - step_t0
                tps      = (tokens_this_step * cfg.log_every_steps) / max(elapsed, 1e-9)
                step_t0  = now
                remaining_steps = cfg.max_steps - self.global_step
                eta      = (elapsed / cfg.log_every_steps) * remaining_steps

                metrics = StepMetrics(
                    step=self.global_step,
                    train_loss=running_loss,
                    val_loss=None,
                    learning_rate=lr,
                    grad_norm=grad_norm,
                    tokens_per_sec=tps,
                    tokens_total=self.tokens_seen,
                    elapsed_sec=now - t0,
                    eta_sec=eta,
                    gpu_mem_gb=get_gpu_mem_gb(),
                    cpu_mem_gb=get_cpu_mem_gb(),
                )
                log.info(str(metrics))

            # ---- Validation ----------------------------------------------- #
            if self.global_step % cfg.val_every_steps == 0:
                val_loss = self._validate()
                if self.is_main:
                    log.info(
                        f"  ↳ Validation  step={self.global_step}  "
                        f"val_loss={val_loss:.6f}"
                    )
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss

                # Save on the validation cadence.
                if self.is_main and self.global_step % cfg.save_every_steps == 0:
                    self.ckpt_manager.save(
                        step=self.global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        model_config=self.model_config,
                        train_config=cfg,
                        train_loss=running_loss,
                        val_loss=val_loss,
                    )
                model.train()

            # ---- Checkpoint (non-validation cadence) ---------------------- #
            elif (
                self.is_main
                and self.global_step % cfg.save_every_steps == 0
            ):
                self.ckpt_manager.save(
                    step=self.global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_config=self.model_config,
                    train_config=cfg,
                    train_loss=running_loss,
                    val_loss=None,
                )

        # Final checkpoint at end of training.
        # _validate() contains a dist.all_reduce, which is a COLLECTIVE: it must
        # be entered by every rank or the ones that skip it leave the others
        # blocked forever. So the validate call sits outside the is_main guard
        # and only the checkpoint write is gated.
        val_loss_final = self._validate()
        if self.is_main:
            self.ckpt_manager.save(
                step=self.global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_config=self.model_config,
                train_config=cfg,
                train_loss=running_loss,
                val_loss=val_loss_final,
            )
            log.info(
                f"Training complete  |  step={self.global_step}  "
                f"final_val_loss={val_loss_final:.6f}"
            )

        if self.distributed:
            dist.destroy_process_group()

    # ---------------------------------------------------------------------- #
    # Internal: validation                                                     #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _validate(self) -> float:
        """
        Run validation over the whole validation set (capped at ``val_steps``
        batches if that is smaller) and return the mean loss.

        The previous implementation always read the first ``val_steps`` windows
        from the start of ``val.bin``, so the tail of the split was never
        evaluated. We now iterate the full loader up to the cap.

        Returns
        -------
        float
            Mean cross-entropy loss over the evaluated validation batches.
        """
        model = self.model
        model.eval()

        total_loss = 0.0
        # 0 or negative ⇒ evaluate the entire validation set.
        max_batches = self.train_config.val_steps
        count      = 0

        for i, (x, y) in enumerate(self.val_loader):
            if max_batches and max_batches > 0 and i >= max_batches:
                break
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                logits, _ = model(x)
                loss = compute_lm_loss(logits, y)
            total_loss += loss.item()
            count += 1

        if count == 0:
            return float("nan")

        mean_loss = total_loss / count

        # Average across DDP ranks.
        if self.distributed:
            loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            mean_loss = loss_tensor.item()

        return mean_loss

    # ---------------------------------------------------------------------- #
    # Internal: resume                                                         #
    # ---------------------------------------------------------------------- #

    def _resume(self, path: Path) -> None:
        """Load a checkpoint and restore training state."""
        log.info(f"Resuming from checkpoint: {path}")
        payload = self.ckpt_manager.load(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.global_step   = payload.get("step", 0)
        self.best_val_loss = payload.get("val_loss") or float("inf")
        # Hand the restored best back to the CheckpointManager. Without this it
        # starts at +inf on every resume, so the first post-resume validation
        # always "improves" and silently overwrites best.pt with a worse model.
        self.ckpt_manager.adopt_state(
            best_val_loss=self.best_val_loss,
            existing_dir=True,
        )
        # tokens_seen: estimate from step count and batch size.
        cfg = self.train_config
        self.tokens_seen = (
            self.global_step
            * cfg.batch_size
            * cfg.seq_len
            * cfg.grad_accum_steps
            * self.world_size
        )
        log.info(
            f"Resumed at step={self.global_step}  "
            f"best_val_loss={self.best_val_loss}"
        )

    # ---------------------------------------------------------------------- #
    # Internal: data iteration                                                 #
    # ---------------------------------------------------------------------- #

    def _new_train_iter(self) -> Iterator:
        return iter(self.train_loader)

    def _next_batch(self, train_iter: Iterator) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(train_iter)
        except StopIteration:
            # One full pass over the sampled windows is exhausted → advance the
            # data epoch so the next pass draws a *different* set of random
            # windows (see BinaryTokenDataset.__getitem__), then rebuild.
            self._data_epoch += 1
            ds = getattr(self.train_loader, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(self._data_epoch)
            self._train_iter = self._new_train_iter()
            return next(self._train_iter)


# --------------------------------------------------------------------------- #
# Null context manager (replaces model.no_sync() on single-GPU)               #
# --------------------------------------------------------------------------- #

class _null_context:
    def __enter__(self): return self
    def __exit__(self, *args): pass
