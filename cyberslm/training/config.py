"""
Training Configuration
=======================
All hyperparameters for a CyberSLM training run, grouped by concern.
Every field has a sensible default calibrated for the 203M-token corpus
on a single GPU with ~24 GB VRAM (e.g., RTX 3090, A10, A100-40GB).

Adjust batch_size, grad_accum_steps, and learning_rate together
to keep the effective batch size and the Adam regime stable.

Effective batch size = batch_size × grad_accum_steps × (world_size for DDP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Complete configuration for one CyberSLM training run."""

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    train_data_path: str = "data/train.bin"
    val_data_path:   str = "data/val.bin"

    # ------------------------------------------------------------------ #
    # Sequences                                                            #
    # ------------------------------------------------------------------ #
    seq_len: int = 4_096   # tokens per sample — matches model max_seq_len

    # ------------------------------------------------------------------ #
    # Batch                                                                #
    # ------------------------------------------------------------------ #
    batch_size: int = 4            # micro-batch per GPU
    grad_accum_steps: int = 16     # effective batch = 4 × 16 = 64 sequences

    # ------------------------------------------------------------------ #
    # Steps                                                                #
    # ------------------------------------------------------------------ #
    # 203 M tokens / (64 seqs × 4096 tokens) ≈ 775 steps per epoch.
    # Train 5 epochs → ~3 875 steps.  Use 4 000 for a round number.
    max_steps: int = 4_000
    warmup_steps: int = 400        # 10 % of max_steps

    # ------------------------------------------------------------------ #
    # Optimiser                                                            #
    # ------------------------------------------------------------------ #
    learning_rate: float  = 3e-4
    min_lr: float         = 3e-5   # cosine decay floor (1/10 of peak)
    weight_decay: float   = 0.1
    beta1: float          = 0.9
    beta2: float          = 0.95
    eps: float            = 1e-8

    # ------------------------------------------------------------------ #
    # Gradient                                                             #
    # ------------------------------------------------------------------ #
    grad_clip: float = 1.0         # max gradient norm before clipping

    # ------------------------------------------------------------------ #
    # Precision                                                            #
    # ------------------------------------------------------------------ #
    # "bfloat16" | "float16" | "float32". bf16 is the right default on any
    # Ampere-or-newer GPU: same exponent range as fp32 so no loss scaling is
    # needed, ~2x the throughput and roughly half the activation memory.
    # Automatically disabled on CPU.
    dtype: str = "bfloat16"

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    val_every_steps: int = 200     # run validation every N steps
    val_steps: int       = 50      # number of validation batches per eval

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #
    checkpoint_dir: str  = "checkpoints"
    save_every_steps: int = 200    # save a checkpoint every N steps
    keep_last_n: int      = 3      # keep the N most recent checkpoints
    save_best: bool       = True   # also save the best-val-loss checkpoint

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    log_every_steps: int = 10      # print metrics every N steps

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Distributed                                                          #
    # ------------------------------------------------------------------ #
    # Set to True to enable DDP.  The training engine reads RANK /
    # LOCAL_RANK / WORLD_SIZE from environment variables (set by torchrun).
    distributed: bool = False

    # ------------------------------------------------------------------ #
    # torch.compile                                                        #
    # ------------------------------------------------------------------ #
    compile_model: bool = False    # set True to enable torch.compile

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    def validate(self) -> "TrainingConfig":
        errors: list[str] = []

        if self.seq_len <= 0:
            errors.append(f"seq_len must be positive, got {self.seq_len}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.grad_accum_steps <= 0:
            errors.append(f"grad_accum_steps must be positive")
        if self.max_steps <= 0:
            errors.append(f"max_steps must be positive")
        if not (0 <= self.warmup_steps <= self.max_steps):
            errors.append(f"warmup_steps must be in [0, max_steps]")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive")
        if not (0.0 < self.min_lr <= self.learning_rate):
            errors.append(f"min_lr must be in (0, learning_rate]")
        if self.grad_clip <= 0:
            errors.append(f"grad_clip must be positive")
        if self.val_steps <= 0:
            errors.append(f"val_steps must be positive")
        if self.keep_last_n <= 0:
            errors.append(f"keep_last_n must be positive")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            errors.append(
                f"dtype must be bfloat16|float16|float32, got {self.dtype!r}"
            )

        if errors:
            raise ValueError(
                "TrainingConfig validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )
        return self

    def __str__(self) -> str:
        eff_batch = self.batch_size * self.grad_accum_steps
        eff_tokens = eff_batch * self.seq_len
        lines = [
            "TrainingConfig",
            "=" * 44,
            f"  train_data_path  : {self.train_data_path}",
            f"  val_data_path    : {self.val_data_path}",
            f"  seq_len          : {self.seq_len:,}",
            f"  batch_size       : {self.batch_size}",
            f"  grad_accum_steps : {self.grad_accum_steps}",
            f"  effective batch  : {eff_batch} seqs  ({eff_tokens:,} tokens)",
            f"  max_steps        : {self.max_steps:,}",
            f"  warmup_steps     : {self.warmup_steps:,}",
            f"  learning_rate    : {self.learning_rate}",
            f"  min_lr           : {self.min_lr}",
            f"  weight_decay     : {self.weight_decay}",
            f"  beta1/beta2      : {self.beta1} / {self.beta2}",
            f"  grad_clip        : {self.grad_clip}",
            f"  dtype            : {self.dtype}",
            f"  val_every_steps  : {self.val_every_steps}",
            f"  checkpoint_dir   : {self.checkpoint_dir}",
            f"  seed             : {self.seed}",
            f"  distributed      : {self.distributed}",
            f"  compile_model    : {self.compile_model}",
        ]
        return "\n".join(lines)
