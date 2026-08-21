"""
CyberSLM SFT — Checkpoint Manager
====================================
Manages saving and loading of all training artifacts needed to:
* Resume a training run without loss of progress.
* Reproduce inference from any saved point.
* Audit the run (config, step, loss) without loading weights.

Checkpoint layout
-----------------

    <output_dir>/
    ├── latest/
    │   ├── model.pt            # model state dict
    │   ├── optimizer.pt        # optimizer state dict
    │   ├── scheduler.pt        # scheduler state dict
    │   ├── training_state.pt   # {epoch, global_step, best_val_loss}
    │   └── sft_config.json     # full SFTConfig snapshot
    ├── best/
    │   └── ...                 # same structure; overwritten on improvement
    └── step_<N>/               # optional periodic versioned snapshots
        └── ...

All ``torch.save`` calls use ``weights_only=False`` for full state dict
compatibility (optimizer / scheduler states include Python objects).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from configs.sft_config import SFTConfig, save_config, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMeta:
    """Lightweight summary of a checkpoint — readable without loading tensors."""
    path:          str
    tag:           str
    epoch:         int
    global_step:   int
    best_val_loss: float
    val_loss:      float   # loss at the time this checkpoint was saved

    def to_dict(self) -> dict:
        return {
            "path":          self.path,
            "tag":           self.tag,
            "epoch":         self.epoch,
            "global_step":   self.global_step,
            "best_val_loss": self.best_val_loss,
            "val_loss":      self.val_loss,
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Handles saving, loading, and bookkeeping of SFT checkpoints.

    Parameters
    ----------
    output_dir:
        Root directory for all checkpoints.  Created if it does not exist.
    keep_last_n:
        How many versioned ``step_<N>`` snapshots to keep on disk.
        Older ones are deleted automatically.  Set to 0 to disable versioned
        snapshots.
    """

    def __init__(self, output_dir: str | Path, keep_last_n: int = 3) -> None:
        self.output_dir  = Path(output_dir)
        self.keep_last_n = keep_last_n
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory history of versioned snapshots (oldest first)
        self._versioned: list[Path] = []

        logger.info("CheckpointManager initialised at: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        tag:       str,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch:     int,
        step:      int,
        val_loss:  float,
        best_val_loss: float,
        cfg:       SFTConfig,
    ) -> Path:
        """
        Save a checkpoint with the given tag.

        Parameters
        ----------
        tag:
            Subdirectory name.  Reserved tags: ``"latest"``, ``"best"``.
            Versioned snapshots use ``"step_<N>"``.
        model / optimizer / scheduler:
            Objects whose state dicts are serialised.
        epoch / step / val_loss / best_val_loss:
            Training metadata written to ``training_state.pt``.
        cfg:
            Master config — written as ``sft_config.json``.

        Returns
        -------
        Path to the checkpoint directory.
        """
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(),     ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        torch.save(
            {
                "epoch":         epoch,
                "global_step":   step,
                "best_val_loss": best_val_loss,
                "val_loss":      val_loss,
            },
            ckpt_dir / "training_state.pt",
        )
        save_config(cfg, ckpt_dir / "sft_config.json")

        # Write a human-readable meta JSON alongside the tensors
        meta = CheckpointMeta(
            path=str(ckpt_dir),
            tag=tag,
            epoch=epoch,
            global_step=step,
            best_val_loss=best_val_loss,
            val_loss=val_loss,
        )
        with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta.to_dict(), fh, indent=2)

        logger.info(
            "Saved checkpoint [%s] epoch=%d step=%d val_loss=%.4f → %s",
            tag, epoch, step, val_loss, ckpt_dir,
        )
        return ckpt_dir

    def save_latest(self, **kwargs) -> Path:
        """Convenience wrapper: save with tag ``"latest"``."""
        return self.save(tag="latest", **kwargs)

    def save_best(self, **kwargs) -> Path:
        """Convenience wrapper: save with tag ``"best"``."""
        return self.save(tag="best", **kwargs)

    def save_versioned(self, step: int, **kwargs) -> Path:
        """
        Save a versioned snapshot (``step_<N>``) and prune older ones.

        Parameters
        ----------
        step:   Current global step — used as the snapshot name.
        **kwargs: Forwarded to ``save()``.
        """
        tag  = f"step_{step:07d}"
        path = self.save(tag=tag, step=step, **kwargs)
        self._versioned.append(path)

        # Prune old versioned checkpoints beyond keep_last_n
        if self.keep_last_n > 0:
            while len(self._versioned) > self.keep_last_n:
                old = self._versioned.pop(0)
                if old.exists():
                    shutil.rmtree(old)
                    logger.debug("Pruned old checkpoint: %s", old)

        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        tag:       str,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device:    torch.device = torch.device("cpu"),
        strict:    bool = True,
    ) -> dict:
        """
        Load a checkpoint by tag.

        Parameters
        ----------
        tag:       Subdirectory tag (e.g. ``"best"``, ``"latest"``).
        model:     Model to load weights into.
        optimizer: If provided, restore optimizer state.
        scheduler: If provided, restore scheduler state.
        device:    Map location for ``torch.load``.
        strict:    Passed to ``model.load_state_dict``.

        Returns
        -------
        dict with keys: ``epoch``, ``global_step``, ``best_val_loss``,
        ``val_loss`` (from the training state).
        """
        ckpt_dir = self.output_dir / tag
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint '{tag}' not found at {ckpt_dir}. "
                "Available: " + ", ".join(self._available_tags())
            )

        # Model weights
        model_path = ckpt_dir / "model.pt"
        if model_path.exists():
            sd = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(sd, strict=strict)
            logger.info("Model weights loaded from %s", model_path)

        # Optimizer
        if optimizer is not None:
            opt_path = ckpt_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device, weights_only=False)
                )
                logger.info("Optimizer state restored from %s", opt_path)

        # Scheduler
        if scheduler is not None:
            sched_path = ckpt_dir / "scheduler.pt"
            if sched_path.exists():
                scheduler.load_state_dict(
                    torch.load(sched_path, map_location=device, weights_only=False)
                )
                logger.info("Scheduler state restored from %s", sched_path)

        # Training state
        ts_path = ckpt_dir / "training_state.pt"
        if ts_path.exists():
            ts = torch.load(ts_path, map_location="cpu", weights_only=False)
        else:
            ts = {"epoch": 0, "global_step": 0, "best_val_loss": float("inf"), "val_loss": float("inf")}

        logger.info(
            "Loaded checkpoint [%s]: epoch=%d step=%d val_loss=%.4f",
            tag, ts.get("epoch", 0), ts.get("global_step", 0), ts.get("val_loss", float("inf")),
        )
        return ts

    def load_latest(self, model: nn.Module, **kwargs) -> dict:
        """Convenience: load the ``"latest"`` checkpoint."""
        return self.load("latest", model=model, **kwargs)

    def load_best(self, model: nn.Module, **kwargs) -> dict:
        """Convenience: load the ``"best"`` checkpoint."""
        return self.load("best", model=model, **kwargs)

    # ------------------------------------------------------------------
    # Pretrained base model loader
    # ------------------------------------------------------------------

    @staticmethod
    def load_pretrained(
        pretrained_path: str | Path,
        model:           nn.Module,
        device:          torch.device,
        strict:          bool = True,
    ) -> None:
        """
        Load the pretrained base model checkpoint to initialise SFT training.

        This is called *once* at the very start of SFT, before any optimizer
        or scheduler state exists.

        Parameters
        ----------
        pretrained_path:
            Path to ``model.pt`` or a directory containing ``model.pt``.
        model:
            Uninitialised CyberSLM model (weights will be overwritten).
        device:
            Map location.
        strict:
            Whether to enforce exact key match.  Should be True for SFT.
        """
        path = Path(pretrained_path)
        if path.is_dir():
            path = path / "model.pt"

        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {path}\n"
                "Set cfg.train.pretrained_checkpoint to the correct path."
            )

        sd = torch.load(path, map_location=device, weights_only=False)
        
        # Stage 1 pretraining checkpoints often contain a full training state dict.
        # If the loaded dict contains "model_state", extract just the model weights.
        if "model_state" in sd:
            sd = sd["model_state"]
            
        model.load_state_dict(sd, strict=strict)
        logger.info(
            "Pretrained base model loaded from %s  "
            "(%.2f M parameters)",
            path,
            sum(p.numel() for p in model.parameters()) / 1e6,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _available_tags(self) -> list[str]:
        return [p.name for p in self.output_dir.iterdir() if p.is_dir()]

    def list_checkpoints(self) -> list[CheckpointMeta]:
        """
        Return metadata for all checkpoints in ``output_dir``, sorted by
        global step (ascending).  Directories without ``meta.json`` are skipped.
        """
        metas: list[CheckpointMeta] = []
        for ckpt_dir in sorted(self.output_dir.iterdir()):
            meta_path = ckpt_dir / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path, encoding="utf-8") as fh:
                d = json.load(fh)
            metas.append(CheckpointMeta(**d))

        metas.sort(key=lambda m: m.global_step)
        return metas

    def best_checkpoint_path(self) -> Optional[Path]:
        """Return the path to the ``best`` checkpoint dir, or None."""
        p = self.output_dir / "best"
        return p if p.exists() else None

    def latest_checkpoint_path(self) -> Optional[Path]:
        """Return the path to the ``latest`` checkpoint dir, or None."""
        p = self.output_dir / "latest"
        return p if p.exists() else None
