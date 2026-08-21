"""
Checkpoint Manager
==================
Handles all checkpoint I/O for CyberSLM training runs.

Checkpoint format
-----------------
Each checkpoint is a single .pt file containing:

    {
        "step":           int,               # optimizer step at save time
        "model_state":    OrderedDict,       # model.state_dict()
        "optimizer_state": dict,             # optimizer.state_dict()
        "scheduler_state": dict,             # scheduler.state_dict()
        "train_loss":     float,             # last training loss
        "val_loss":       float | None,      # validation loss (if available)
        "config":         dict,              # CyberSLMConfig as plain dict
        "train_config":   dict,             # TrainingConfig as plain dict
        "rng_state":      dict,              # torch / numpy / python RNG state
    }

Naming convention
-----------------
  checkpoints/ckpt_step_{step:07d}.pt    — periodic checkpoint
  checkpoints/best.pt                     — best validation loss checkpoint
  checkpoints/latest.txt                  — plaintext path of most recent ckpt

Rotation
--------
Only the last ``keep_last_n`` periodic checkpoints are kept on disk.
``best.pt`` is never rotated.
"""

from __future__ import annotations

import math

import json
import logging
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cyberslm.model.config import CyberSLMConfig
from cyberslm.training.config import TrainingConfig
from cyberslm.training.scheduler import CosineWarmupScheduler

log = logging.getLogger(__name__)


class CheckpointManager:
    """
    Save and restore training checkpoints.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory where checkpoints are written.
    keep_last_n : int
        Number of most-recent periodic checkpoints to retain.
    save_best : bool
        Whether to maintain a ``best.pt`` checkpoint tracking minimum val loss.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_last_n: int = 3,
        save_best: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n    = keep_last_n
        self.save_best      = save_best
        self._saved: List[Path] = []   # ordered list of periodic ckpt paths
        self._best_val_loss: float = float("inf")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Resume support                                                           #
    # ---------------------------------------------------------------------- #

    def adopt_state(
        self,
        best_val_loss: float,
        existing_dir: bool = True,
    ) -> None:
        """
        Re-seed the manager's in-memory state after a resume.

        Two pieces of state live only in this object and are otherwise lost when
        training restarts from a checkpoint:

        ``_best_val_loss``
            Starts at +inf on a fresh manager. Left alone, the first validation
            after a resume beats +inf unconditionally and overwrites ``best.pt``
            with whatever the current (possibly worse) model is.

        ``_saved``
            Starts empty, so ``_rotate()`` has no idea that older checkpoints
            already exist on disk and never prunes them.
        """
        if best_val_loss is not None and math.isfinite(best_val_loss):
            self._best_val_loss = float(best_val_loss)
            log.info("CheckpointManager: best_val_loss restored to %.6f", best_val_loss)

        if existing_dir and self.checkpoint_dir.exists():
            self._saved = sorted(self.checkpoint_dir.glob("ckpt_step_*.pt"))
            if self._saved:
                log.info(
                    "CheckpointManager: adopted %d existing checkpoint(s) for rotation",
                    len(self._saved),
                )

    # ---------------------------------------------------------------------- #
    # Save                                                                     #
    # ---------------------------------------------------------------------- #

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: CosineWarmupScheduler,
        model_config: CyberSLMConfig,
        train_config: TrainingConfig,
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> Path:
        """
        Save a checkpoint and rotate old ones.

        Parameters
        ----------
        step : int
            Current optimizer step.
        model : nn.Module
            The model (may be wrapped in DDP — unwrapped automatically).
        optimizer : optim.Optimizer
        scheduler : CosineWarmupScheduler
        model_config : CyberSLMConfig
        train_config : TrainingConfig
        train_loss : float
        val_loss : float, optional

        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        # DDP wraps the model; always save the underlying module.
        raw_model = model.module if hasattr(model, "module") else model

        payload: Dict = {
            "step":             step,
            "model_state":      raw_model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "train_loss":       train_loss,
            "val_loss":         val_loss,
            "config":           asdict(model_config),
            "train_config":     asdict(train_config),
            "rng_state":        self._get_rng_state(),
        }

        ckpt_path = self.checkpoint_dir / f"ckpt_step_{step:07d}.pt"
        # Atomic write: write to a temp file then rename.
        tmp_path = ckpt_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)   # atomic on POSIX *and* Windows

        log.info(f"Checkpoint saved → {ckpt_path}  (val_loss={val_loss})")

        # Update latest pointer.
        self._write_latest(ckpt_path)

        # Rotate old checkpoints.
        self._saved.append(ckpt_path)
        self._rotate()

        # Save best.
        if self.save_best and val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best.pt"
            tmp_best  = best_path.with_suffix(".tmp")
            # Copy the bytes we already wrote rather than re-serialising the
            # whole payload a second time (~400 MB of avoidable I/O per save).
            shutil.copyfile(ckpt_path, tmp_best)
            os.replace(tmp_best, best_path)
            log.info(f"New best checkpoint -> {best_path}  (val_loss={val_loss:.6f})")

        return ckpt_path

    # ---------------------------------------------------------------------- #
    # Load                                                                     #
    # ---------------------------------------------------------------------- #

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[CosineWarmupScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> Dict:
        """
        Restore a checkpoint into model (and optionally optimizer/scheduler).

        Parameters
        ----------
        path : str | Path
            Path to the ``.pt`` checkpoint file.
        model : nn.Module
            Target model (DDP-unwrapped automatically).
        optimizer : Optimizer, optional
            If provided, optimizer state is restored.
        scheduler : CosineWarmupScheduler, optional
            If provided, scheduler state is restored.
        device : torch.device, optional
            Map location for tensor loading.

        Returns
        -------
        dict
            The raw checkpoint dictionary (contains ``step``, ``val_loss``, etc.)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = device if device is not None else "cpu"
        payload = torch.load(path, map_location=map_location, weights_only=False)

        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(payload["model_state"])

        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])

        if scheduler is not None and "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])

        if "rng_state" in payload:
            self._set_rng_state(payload["rng_state"])

        log.info(
            f"Checkpoint loaded ← {path}  "
            f"(step={payload.get('step')}, val_loss={payload.get('val_loss')})"
        )
        return payload

    # ---------------------------------------------------------------------- #
    # Discovery                                                                #
    # ---------------------------------------------------------------------- #

    def latest_checkpoint(self) -> Optional[Path]:
        """Return the path of the most recent checkpoint, or None."""
        latest_txt = self.checkpoint_dir / "latest.txt"
        if latest_txt.exists():
            candidate = Path(latest_txt.read_text().strip())
            if candidate.exists():
                return candidate
        # Fall back to scanning the directory.
        ckpts = sorted(self.checkpoint_dir.glob("ckpt_step_*.pt"))
        return ckpts[-1] if ckpts else None

    def best_checkpoint(self) -> Optional[Path]:
        """Return the path of the best checkpoint, or None."""
        p = self.checkpoint_dir / "best.pt"
        return p if p.exists() else None

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _rotate(self) -> None:
        """Delete checkpoints beyond keep_last_n."""
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()
                log.debug(f"Rotated old checkpoint: {old}")

    def _write_latest(self, path: Path) -> None:
        latest_txt = self.checkpoint_dir / "latest.txt"
        latest_txt.write_text(str(path))

    @staticmethod
    def _get_rng_state() -> Dict:
        state: Dict = {
            "torch":  torch.get_rng_state(),
            "python": random.getstate(),
        }
        try:
            import numpy as np
            state["numpy"] = np.random.get_state()
        except ImportError:
            pass
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _as_cpu_byte(t):
        """
        Coerce a saved RNG blob back to the CPU uint8 tensor torch demands.

        ``load()`` calls ``torch.load(..., map_location=device)``. When device is
        CUDA that moves EVERY tensor in the payload to the GPU, including the RNG
        state, and ``torch.set_rng_state`` then rejects it with
        "RNG state must be a torch.ByteTensor". This made every GPU resume fail
        on the first checkpoint it tried to load.
        """
        if torch.is_tensor(t):
            return t.detach().to(device="cpu", dtype=torch.uint8)
        return t

    @classmethod
    def _set_rng_state(cls, state: Dict) -> None:
        # RNG restore is a nice-to-have for reproducibility, never a reason to
        # lose a run: a resume must not die because a generator state is stale
        # or came from a different device/torch build.
        try:
            if "torch" in state:
                torch.set_rng_state(cls._as_cpu_byte(state["torch"]))
            if "python" in state:
                random.setstate(state["python"])
            try:
                import numpy as np
                if "numpy" in state:
                    np.random.set_state(state["numpy"])
            except ImportError:
                pass
            if "cuda" in state and torch.cuda.is_available():
                cuda_states = [cls._as_cpu_byte(t) for t in state["cuda"]]
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)
                else:
                    log.warning(
                        "Checkpoint holds %d CUDA RNG states but this host has %d "
                        "device(s); skipping CUDA RNG restore.",
                        len(cuda_states), torch.cuda.device_count(),
                    )
        except Exception as exc:  # noqa: BLE001 - deliberately non-fatal
            log.warning("Could not restore RNG state (%s); continuing with a "
                        "fresh generator. Training is unaffected apart from "
                        "exact data-order reproducibility.", exc)
