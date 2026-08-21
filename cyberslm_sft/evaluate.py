"""
CyberSLM SFT — Evaluation Script
===================================
Standalone evaluation script that:
1. Loads the best (or specified) SFT checkpoint.
2. Runs the full validation loop.
3. Prints a detailed metric report.
4. Optionally saves the report to a JSON file.

Usage::

    python evaluate.py \\
        --checkpoint checkpoints/best \\
        --config     checkpoints/best/sft_config.json \\
        --data       data/val.jsonl \\
        --output     eval_results.json

This script is intentionally decoupled from the trainer so it can be run
on any checkpoint, including those from other machines or earlier runs.
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
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.sft_config import SFTConfig, load_config, default_config
from data.collator import SFTCollator
from data.dataset_loader import load_jsonl
from data.dataset_validator import validate_dataset
from data.prompt_formatter import PromptFormatter, Tokenizer
from data.sft_dataset import SFTDataset
from utils.checkpoint_manager import CheckpointManager
from utils.logging_utils import setup_logging
from utils.validation import run_validation
from utils.seed import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric report
# ---------------------------------------------------------------------------

def build_eval_report(
    val_result,
    cfg:         SFTConfig,
    checkpoint:  str,
    n_samples:   int,
    elapsed:     float,
    device:      str = "cpu",
) -> dict:
    """Assemble a structured evaluation report dict."""
    return {
        "checkpoint":   checkpoint,
        "model": {
            "hidden_size":   cfg.model.hidden_size,
            "num_layers":    cfg.model.num_layers,
            "vocab_size":    cfg.model.vocab_size,
        },
        "dataset": {
            "val_samples":   n_samples,
            "max_seq_len":   cfg.data.max_seq_len,
        },
        "metrics": {
            "val_loss":      round(val_result.val_loss, 6),
            "val_ppl":       round(val_result.val_ppl, 4),
            "tok_per_sec":   round(val_result.tok_per_sec, 1),
            "tokens_seen":   val_result.tokens_seen,
            "n_batches":     val_result.n_batches,
        },
        "run": {
            "elapsed_s":     round(elapsed, 2),
            "device":        device,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    config_path:     str,
    data_path:       str,
    output_path:     str | None = None,
    batch_size:      int = 8,
    max_samples:     int = -1,
    device_str:      str = "cpu",
) -> dict:
    """
    Run full evaluation.

    Parameters
    ----------
    checkpoint_path: Path to the checkpoint directory (containing ``model.pt``).
    config_path:     Path to ``sft_config.json``.
    data_path:       Path to the validation ``.jsonl`` file.
    output_path:     If set, write the JSON report here.
    batch_size:      Evaluation batch size.
    max_samples:     Cap on validation samples (-1 = no cap).
    device_str:      Device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    Evaluation report dict.
    """
    setup_logging(log_level=logging.INFO)
    device = torch.device(device_str)

    # --- Config ---
    if Path(config_path).exists():
        cfg = load_config(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        logger.warning("Config not found at %s — using defaults", config_path)
        cfg = default_config()

    set_seed(cfg.train.seed)

    # --- Tokenizer ---
    tokenizer = Tokenizer(cfg.tokenizer.model_path)

    # --- Model ---
    model = _build_model(cfg, device)
    ckpt_mgr = CheckpointManager(output_dir=str(Path(checkpoint_path).parent))
    ckpt_tag  = Path(checkpoint_path).name
    ckpt_mgr.load(
        tag=ckpt_tag,
        model=model,
        device=device,
        strict=True,
    )
    model.eval()

    # --- Data ---
    raw_samples = load_jsonl(data_path, max_samples=max_samples)
    valid_samples, _ = validate_dataset(raw_samples, strict=False)

    formatter  = PromptFormatter(cfg=cfg, tokenizer=tokenizer)
    val_ds     = SFTDataset(valid_samples, formatter, split="eval")
    collator   = SFTCollator(pad_id=tokenizer.pad_id, max_seq_len=cfg.data.max_seq_len)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    logger.info("Evaluating %d samples on %s ...", len(val_ds), device)

    # --- Evaluate ---
    t_start = time.time()
    result  = run_validation(model=model, val_loader=val_loader, device=device)
    elapsed = time.time() - t_start

    # --- Report ---
    report = build_eval_report(
        val_result=result,
        cfg=cfg,
        checkpoint=str(checkpoint_path),
        n_samples=len(val_ds),
        elapsed=elapsed,
        device=device_str,
    )

    _print_report(report)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Evaluation report saved to %s", output_path)

    return report


def _print_report(report: dict) -> None:
    m = report["metrics"]
    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info("  Checkpoint : %s", report["checkpoint"])
    logger.info("  Val Loss   : %.4f", m["val_loss"])
    logger.info("  Perplexity : %.2f", m["val_ppl"])
    logger.info("  Tok/sec    : %d", int(m["tok_per_sec"]))
    logger.info("  Tokens     : %d", m["tokens_seen"])
    logger.info("  Batches    : %d", m["n_batches"])
    logger.info("  Elapsed    : %.1fs", report["run"]["elapsed_s"])
    logger.info("=" * 50)


def _build_model(cfg: SFTConfig, device: torch.device):
    """
    Build the CyberSLM model from config.

    This imports the model architecture module.  The architecture file must
    be present at ``model/cyberslm.py`` (from Stage 1 pretraining).
    We fall back gracefully with a clear error if it is missing.
    """
    try:
        from model.cyberslm import CyberSLM
        return CyberSLM(cfg.model).to(device)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Cannot import CyberSLM model architecture.\n"
            "Ensure 'model/cyberslm.py' exists (from Stage 1 pretraining).\n"
            "The SFT pipeline does not redefine the architecture."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CyberSLM-Instruct checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint directory (e.g. checkpoints/best)")
    p.add_argument("--config",     required=True,
                   help="Path to sft_config.json")
    p.add_argument("--data",       required=True,
                   help="Path to validation JSONL file")
    p.add_argument("--output",     default=None,
                   help="Path to write the JSON report (optional)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device_str=args.device,
    )
