"""
CyberSLM SFT — Training Entry Point
=====================================
Wires every pipeline stage together and launches instruction fine-tuning.

Usage
-----

Basic run::

    python train.py

With custom config overrides::

    python train.py \\
        --pretrained  checkpoints/pretrained/model.pt \\
        --train-data  data/train.jsonl \\
        --val-data    data/val.jsonl \\
        --tokenizer   tokenizer/tokenizer.model \\
        --output-dir  checkpoints \\
        --epochs      3 \\
        --lr          2e-5 \\
        --batch-size  4 \\
        --accum-steps 4

Resume a run::

    python train.py --resume checkpoints/latest

All arguments override the defaults in ``SFTConfig``.
"""

from __future__ import annotations

# --- path bootstrap: allow running this file from any working directory ---
import sys as _sys
from pathlib import Path as _Path
_SFT_ROOT = _Path(__file__).resolve().parent
if str(_SFT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SFT_ROOT))
# -------------------------------------------------------------------------

import argparse
import logging
import sys
from pathlib import Path

import torch

from configs.sft_config import SFTConfig, default_config, save_config, load_config
from data.prompt_formatter import Tokenizer
from data.sft_dataset import build_datasets
from trainer import SFTTrainer
from utils.checkpoint_manager import CheckpointManager
from utils.logging_utils import setup_logging
from utils.seed import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(cfg: SFTConfig, device: torch.device):
    """
    Import and instantiate the CyberSLM architecture from Stage 1.

    The architecture file must be at ``model/cyberslm.py``.  The SFT
    pipeline never re-defines the model — it only imports it.
    """
    try:
        from model.cyberslm import CyberSLM
    except ModuleNotFoundError:
        logger.error(
            "Cannot find 'model/cyberslm.py'. "
            "Ensure the Stage 1 pretraining model file is present. "
            "The SFT pipeline does not redefine the architecture."
        )
        sys.exit(1)

    model = CyberSLM(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model built: %.2f M parameters on %s", n_params, device)
    return model


# ---------------------------------------------------------------------------
# Architecture inheritance
# ---------------------------------------------------------------------------

_ARCH_MAP = {
    "vocab_size": "vocab_size",
    "hidden_dim": "hidden_size",
    "num_layers": "num_layers",
    "num_heads":  "num_heads",
    "head_dim":   "head_dim",
    "ffn_hidden_dim": "ffn_size",
    "max_seq_len": "max_seq_len",
    "norm_eps":    "norm_eps",
    "tie_weights": "weight_tying",
}


def _inherit_arch_from_checkpoint(cfg: SFTConfig) -> None:
    """Overwrite cfg.model from the pretrained checkpoint's saved config."""
    path = Path(cfg.train.pretrained_checkpoint)
    if path.is_dir():
        path = path / "model.pt"
    if not path.exists():
        logger.warning("Pretrained checkpoint %s not found yet; keeping configured architecture.", path)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("config") if isinstance(payload, dict) else None
    if not saved:
        logger.info("Checkpoint carries no config block; keeping configured architecture.")
        return
    changed = []
    for ck, mk in _ARCH_MAP.items():
        if ck in saved and hasattr(cfg.model, mk):
            old = getattr(cfg.model, mk)
            new = saved[ck]
            if old != new:
                changed.append(f"{mk}: {old} -> {new}")
            setattr(cfg.model, mk, new)
    if changed:
        logger.info("Architecture inherited from checkpoint: %s", "; ".join(changed))
    else:
        logger.info("Architecture already matches the checkpoint.")
    if cfg.data.max_seq_len > cfg.model.max_seq_len:
        logger.warning(
            "data.max_seq_len (%d) exceeds the base model's trained context (%d); clamping.",
            cfg.data.max_seq_len, cfg.model.max_seq_len,
        )
        cfg.data.max_seq_len = cfg.model.max_seq_len


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        logger.info(
            "Using CUDA: %s  (%.1f GB VRAM)",
            props.name,
            props.total_memory / 1e9,
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: SFTConfig) -> None:
    """
    Run the full SFT pipeline:
    1. Set seed
    2. Build tokenizer
    3. Build & load pretrained model
    4. Build datasets
    5. Instantiate trainer
    6. Train
    """
    setup_logging(
        log_file=str(Path(cfg.train.output_dir) / "train.log"),
        log_level=logging.INFO,
    )

    set_seed(cfg.train.seed)
    device = _select_device()

    # ---- Tokenizer ----
    logger.info("Loading tokenizer from %s", cfg.tokenizer.model_path)
    tokenizer = Tokenizer(cfg.tokenizer.model_path)
    logger.info("Tokenizer loaded: %d tokens", len(tokenizer))

    # ---- Inherit architecture from the pretrained checkpoint ----
    # The SFT ModelConfig has to match the base model exactly. Hardcoding it
    # means a base trained at a different width/depth/context silently loads
    # into a mismatched shell (RoPE buffers are non-persistent, so a max_seq_len
    # mismatch would not even raise). Read the truth off the checkpoint.
    _inherit_arch_from_checkpoint(cfg)

    # ---- Model ----
    model = _build_model(cfg, device)

    # ---- Load pretrained weights ----
    CheckpointManager.load_pretrained(
        pretrained_path=cfg.train.pretrained_checkpoint,
        model=model,
        device=device,
        strict=True,
    )

    # ---- Datasets ----
    logger.info("Building datasets …")
    train_ds, val_ds = build_datasets(cfg, tokenizer)
    logger.info(
        "Datasets ready: %d train | %d val", len(train_ds), len(val_ds)
    )

    # ---- Save config snapshot at run start ----
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    save_config(cfg, Path(cfg.train.output_dir) / "sft_config.json")

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        device=device,
        tokenizer=tokenizer,
    )

    # ---- Train ----
    trainer.train()

    logger.info(
        "SFT complete. Best checkpoint: %s/best",
        cfg.train.output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CyberSLM Supervised Instruction Fine-Tuning"
    )
    p.add_argument("--pretrained",   default=None,
                   help="Path to pretrained base model checkpoint (model.pt or dir)")
    p.add_argument("--train-data",   default=None,
                   help="Path to training JSONL file")
    p.add_argument("--val-data",     default=None,
                   help="Path to validation JSONL file (optional; auto-split if absent)")
    p.add_argument("--tokenizer",    default=None,
                   help="Path to SentencePiece tokenizer model")
    p.add_argument("--output-dir",   default=None,
                   help="Directory for checkpoints and logs")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None,
                   help="Peak learning rate (default: 2e-5)")
    p.add_argument("--batch-size",   type=int,   default=None,
                   help="Per-device batch size")
    p.add_argument("--accum-steps",  type=int,   default=None,
                   help="Gradient accumulation steps")
    p.add_argument("--max-seq-len",  type=int,   default=None,
                   help="Maximum token sequence length")
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--resume",       default=None,
                   help="Path to checkpoint directory to resume from")
    p.add_argument("--config",       default=None,
                   help="Path to a saved sft_config.json to load as base config")
    return p.parse_args()


def _apply_overrides(cfg: SFTConfig, args: argparse.Namespace) -> SFTConfig:
    """Apply CLI argument overrides on top of the loaded/default config."""
    if args.pretrained:
        cfg.train.pretrained_checkpoint = args.pretrained
    if args.train_data:
        cfg.data.train_path = args.train_data
    if args.val_data:
        cfg.data.val_path = args.val_data
    if args.tokenizer:
        cfg.tokenizer.model_path = args.tokenizer
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.train.per_device_batch_size = args.batch_size
    if args.accum_steps is not None:
        cfg.train.gradient_accumulation_steps = args.accum_steps
    if args.max_seq_len is not None:
        cfg.data.max_seq_len = args.max_seq_len
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.resume:
        cfg.train.resume_from_checkpoint = args.resume
    return cfg


if __name__ == "__main__":
    args = _parse_args()

    # Load base config from file if provided, else use defaults
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    cfg = _apply_overrides(cfg, args)
    main(cfg)
