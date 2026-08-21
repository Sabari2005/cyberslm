"""
CyberSLM SFT — Dataset Loader
==============================
Loads, merges, and splits JSONL instruction datasets.

Supported sample formats
------------------------

**Alpaca format**::

    {"instruction": "...", "input": "...", "output": "..."}

    ``input`` is optional; omit or set to empty string when not needed.

**Conversation format**::

    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

    Multi-turn conversations are supported.  The loader preserves the full
    message list so the formatter can build the correct token sequence.

Both formats may be mixed inside the same JSONL file.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from configs.sft_config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

# Normalised in-memory representation used by the rest of the pipeline.
# Every raw sample is converted to one of these two shapes.

AlpacaSample = dict  # keys: instruction, input (may be ""), output
ConvSample = dict    # keys: messages  (list of {role, content} dicts)

RawSample = AlpacaSample | ConvSample


# ---------------------------------------------------------------------------
# Low-level JSONL reader
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict]:
    """
    Yield parsed JSON objects from a JSONL file one line at a time.
    Blank lines and comment lines (starting with ``#``) are skipped.
    Malformed lines emit a warning and are skipped to avoid crashing a run.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON at %s:%d — %s", path.name, lineno, exc
                )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_conversation(sample: dict) -> bool:
    """Return True if the sample uses the conversation (messages) format."""
    return "messages" in sample


def _is_alpaca(sample: dict) -> bool:
    """Return True if the sample uses the alpaca (instruction/output) format."""
    return "instruction" in sample and "output" in sample


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(sample: dict) -> Optional[RawSample]:
    """
    Accept a raw parsed dict and return a canonical form, or None if the
    sample cannot be recognised as either supported format.
    """
    if _is_conversation(sample):
        messages: list = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            return None
        # Ensure every message has the required keys
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return None
        return {"messages": messages}

    if _is_alpaca(sample):
        return {
            "instruction": str(sample.get("instruction", "")).strip(),
            "input":       str(sample.get("input", "")).strip(),
            "output":      str(sample.get("output", "")).strip(),
        }

    return None


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_jsonl(
    path: str | Path,
    max_samples: int = -1,
) -> List[RawSample]:
    """
    Load and normalise all samples from a single JSONL file.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file.
    max_samples:
        If > 0, stop after this many *valid* samples have been loaded.
        Useful for smoke-tests (``DataConfig.max_samples``).

    Returns
    -------
    list[RawSample]
        Normalised sample dicts ready for validation and formatting.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: List[RawSample] = []
    skipped = 0

    for raw in _iter_jsonl(path):
        normalised = _normalise(raw)
        if normalised is None:
            skipped += 1
            continue
        samples.append(normalised)
        if max_samples > 0 and len(samples) >= max_samples:
            break

    logger.info(
        "Loaded %d samples from %s  (skipped %d unrecognised)",
        len(samples), path.name, skipped,
    )
    return samples


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def _split_dataset(
    samples: List[RawSample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[RawSample], List[RawSample]]:
    """
    Deterministic random split.  Shuffles a *copy* so the original order is
    not mutated.

    Parameters
    ----------
    samples:   Full dataset.
    val_ratio: Fraction used for validation (e.g. 0.05 → 5 %).
    seed:      RNG seed for reproducibility.

    Returns
    -------
    (train_samples, val_samples)
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = set(indices[:n_val])

    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val   = [s for i, s in enumerate(samples) if i in val_idx]

    logger.info(
        "Split: %d train / %d validation  (val_ratio=%.3f, seed=%d)",
        len(train), len(val), val_ratio, seed,
    )
    return train, val


# ---------------------------------------------------------------------------
# Public API — load_datasets
# ---------------------------------------------------------------------------

def load_datasets(
    cfg: DataConfig,
) -> Tuple[List[RawSample], List[RawSample]]:
    """
    Entry point used by the trainer.

    Logic
    -----
    1. Load training data from ``cfg.train_path``.
    2. If ``cfg.val_path`` points to an existing file, load it separately.
    3. Otherwise, carve out ``cfg.val_split`` fraction from training data.

    Parameters
    ----------
    cfg:
        ``DataConfig`` from the master ``SFTConfig``.

    Returns
    -------
    (train_samples, val_samples)
        Both lists contain normalised ``RawSample`` dicts.
    """
    train_path = Path(cfg.train_path)
    val_path   = Path(cfg.val_path) if cfg.val_path else None

    # Load training data
    train_samples = load_jsonl(train_path, max_samples=cfg.max_samples)

    # Load / derive validation data
    if val_path and val_path.exists():
        val_max = max(1, int(cfg.max_samples * cfg.val_split)) if cfg.max_samples > 0 else -1
        val_samples = load_jsonl(val_path, max_samples=val_max)
        logger.info(
            "Using dedicated validation file: %s (%d samples)",
            val_path.name, len(val_samples),
        )
    else:
        if val_path:
            logger.warning(
                "val_path '%s' does not exist — falling back to auto-split.", val_path
            )
        train_samples, val_samples = _split_dataset(
            train_samples, val_ratio=cfg.val_split, seed=cfg.seed
        )

    return train_samples, val_samples


# ---------------------------------------------------------------------------
# Convenience: quick dataset stats
# ---------------------------------------------------------------------------

def dataset_stats(samples: List[RawSample]) -> dict:
    """
    Return a summary dict with counts and format breakdown.
    Useful for logging at startup.
    """
    n_alpaca = sum(1 for s in samples if _is_alpaca(s))
    n_conv   = sum(1 for s in samples if _is_conversation(s))

    stats = {
        "total":        len(samples),
        "alpaca_format": n_alpaca,
        "conv_format":   n_conv,
    }
    return stats
