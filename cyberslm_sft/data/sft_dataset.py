"""
CyberSLM SFT — SFT Dataset (torch.utils.data.Dataset)
=======================================================
Wraps a pre-formatted list of ``(input_ids, labels)`` pairs as a
``torch.utils.data.Dataset`` for use with ``DataLoader``.

Pre-tokenisation strategy
--------------------------
All samples are tokenised **once** at construction time (not lazily) so that:

* The DataLoader workers never call the tokeniser — no SentencePiece fork
  issues on Linux.
* Per-epoch shuffling is handled by the ``DataLoader``'s sampler, not here.
* ``__len__`` and ``__getitem__`` are O(1).

For very large datasets (>10 M samples) consider streaming instead; for the
scale CyberSLM is designed for (< 1 M samples) this is fine.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from configs.sft_config import SFTConfig
from data.dataset_loader import load_datasets, dataset_stats
from data.dataset_validator import validate_dataset
from data.prompt_formatter import IGNORE_INDEX, PromptFormatter, Tokenizer

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """
    Pre-tokenised instruction fine-tuning dataset.

    Parameters
    ----------
    samples:
        Normalised raw samples (output of ``dataset_loader``).
    formatter:
        Initialised ``PromptFormatter`` instance.
    split:
        ``"train"`` or ``"val"`` — used only for logging.
    """

    def __init__(
        self,
        samples:   List[dict],
        formatter: PromptFormatter,
        split:     str = "train",
    ) -> None:
        self.split = split
        self._data: List[Tuple[List[int], List[int]]] = []

        skipped = 0
        for raw in samples:
            result = formatter.format(raw)
            if result is None:
                skipped += 1
                continue
            self._data.append(result)

        logger.info(
            "[%s] SFTDataset: %d samples tokenised, %d skipped (no response tokens)",
            split, len(self._data), skipped,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self._data[idx]

    # ------------------------------------------------------------------
    def response_token_stats(self) -> dict:
        """Return basic statistics about response token density."""
        if not self._data:
            return {"samples": 0}

        total_tokens   = 0
        active_tokens  = 0
        min_active     = float("inf")
        max_active     = 0

        for _, labels in self._data:
            total = len(labels)
            active = sum(1 for l in labels if l != IGNORE_INDEX)
            total_tokens  += total
            active_tokens += active
            min_active = min(min_active, active)
            max_active = max(max_active, active)

        return {
            "samples":       len(self._data),
            "total_tokens":  total_tokens,
            "active_tokens": active_tokens,
            "active_ratio":  active_tokens / max(total_tokens, 1),
            "min_active":    int(min_active) if min_active != float("inf") else 0,
            "max_active":    max_active,
            "avg_active":    active_tokens / max(len(self._data), 1),
        }


# ---------------------------------------------------------------------------
# Factory — build both splits from config
# ---------------------------------------------------------------------------

def build_datasets(
    cfg:       SFTConfig,
    tokenizer: Tokenizer,
) -> Tuple[SFTDataset, SFTDataset]:
    """
    Load → validate → format → wrap as SFTDataset for both splits.

    This is the single entry point used by the trainer.

    Parameters
    ----------
    cfg:       Master ``SFTConfig``.
    tokenizer: Initialised ``Tokenizer`` for the SentencePiece model.

    Returns
    -------
    (train_dataset, val_dataset)
    """
    # 1. Load
    train_raw, val_raw = load_datasets(cfg.data)

    logger.info("Train raw stats: %s", dataset_stats(train_raw))
    logger.info("Val   raw stats: %s", dataset_stats(val_raw))

    # 2. Validate
    train_raw, train_report = validate_dataset(train_raw, strict=False)
    val_raw,   val_report   = validate_dataset(val_raw,   strict=False)

    if len(train_raw) == 0:
        raise RuntimeError(
            "Training dataset is empty after validation. "
            "Check your dataset format and paths."
        )

    # 3. Formatter
    formatter = PromptFormatter(cfg=cfg, tokenizer=tokenizer)

    # 4. Wrap
    train_ds = SFTDataset(train_raw, formatter, split="train")
    val_ds   = SFTDataset(val_raw,   formatter, split="val")

    # 5. Log response token density
    train_stats = train_ds.response_token_stats()
    logger.info(
        "Train token stats: %d samples | %.1f %% active tokens | "
        "avg %.1f response tokens/sample",
        train_stats["samples"],
        100.0 * train_stats["active_ratio"],
        train_stats["avg_active"],
    )

    return train_ds, val_ds
