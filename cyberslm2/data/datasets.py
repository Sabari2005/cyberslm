"""
datasets.py
===========
Pretraining and SFT datasets.

PackedPretrainDataset walks the token stream in **non-overlapping** windows and
shuffles the order of those windows each epoch. The v1 model instead drew random
start offsets with replacement, which has two defects: tokens near the middle of
the file are oversampled relative to the edges, and with replacement roughly
1/e ~ 37% of positions are never visited in a given pass. Sequential windows see
every token exactly once per epoch, which matters when the corpus is the scarce
resource.
"""

from __future__ import annotations

import json
import mmap
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cyberslm2.data.special_tokens import (
    ASSISTANT_ID,
    BOS_ID,
    END_ID,
    EOS_ID,
    IGNORE_INDEX,
    SYSTEM_ID,
    THINK_END_ID,
    THINK_ID,
    TOOL_CALL_ID,
    TOOL_END_ID,
    TOOL_LIST_ID,
    TOOL_RESULT_ID,
    USER_ID,
)

BYTES_PER_TOKEN = 2  # uint16


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

class PackedPretrainDataset(Dataset):
    """
    Non-overlapping windows over a flat uint16 token file.

    Each item is a (seq_len + 1,) slice; the trainer shifts it into input/target.
    The extra +1 token means consecutive windows share one boundary token, which
    is what makes the target for the last real position well defined.
    """

    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int,
        seed: int = 1337,
        shuffle: bool = True,
    ) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Token file not found: {self.bin_path}. "
                "Run the preprocessing pipeline first."
            )

        self.seq_len = seq_len
        self.seed = seed
        self.shuffle = shuffle

        n_bytes = self.bin_path.stat().st_size
        if n_bytes % BYTES_PER_TOKEN:
            raise ValueError(
                f"{self.bin_path} size {n_bytes} is not a multiple of "
                f"{BYTES_PER_TOKEN}; the file looks truncated."
            )
        self.n_tokens = n_bytes // BYTES_PER_TOKEN
        self.n_windows = max(0, (self.n_tokens - 1) // seq_len)
        if self.n_windows == 0:
            raise ValueError(
                f"{self.bin_path} holds {self.n_tokens} tokens, too few for a "
                f"single {seq_len}-token window."
            )

        # Opened lazily: DataLoader workers on Windows use spawn, which pickles
        # the dataset, and neither mmap nor a file handle is picklable.
        self._fh = None
        self._mm = None

        self._epoch = 0
        self._order = np.arange(self.n_windows)
        if shuffle:
            self.set_epoch(0)

    # -- process-safety -----------------------------------------------------

    def _open(self) -> None:
        if self._mm is None:
            self._fh = self.bin_path.open("rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_mm"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        try:
            if self._mm is not None:
                self._mm.close()
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass

    # -- epochs -------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle window order. Call once per epoch, before iterating."""
        self._epoch = epoch
        if self.shuffle:
            rng = np.random.default_rng([self.seed, epoch])
            self._order = rng.permutation(self.n_windows)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tensor:
        self._open()
        w = int(self._order[idx])
        start = w * self.seq_len
        byte_start = start * BYTES_PER_TOKEN
        byte_end = byte_start + (self.seq_len + 1) * BYTES_PER_TOKEN

        chunk = np.frombuffer(self._mm[byte_start:byte_end], dtype="<u2")
        return torch.from_numpy(chunk.astype(np.int64))


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def _encode(sp, text: str) -> list[int]:
    return sp.encode(text, out_type=int)


def build_sft_example(
    sample: dict[str, Any],
    sp,
    max_seq_len: int,
    train_on_reasoning: bool = True,
) -> Optional[dict[str, list[int]]]:
    """
    Turn one JSONL record into (input_ids, labels).

    Accepts both shapes found in the existing SFT file:
      {"instruction": ..., "input": ..., "output": ...}
      {"messages": [{"role": "user"|"assistant"|"system"|"tool", "content": ...}]}

    Labels are built from the *same* token lists that build input_ids, segment by
    segment. No character offsets, no re-encoding of prefixes: those drift the
    moment the tokenizer adds a dummy space prefix, and a drifted boundary
    silently trains the model on its own prompt.

    Returns None if the example produces no trainable token.
    """
    segments: list[tuple[list[int], bool]] = []   # (ids, is_loss_target)

    def add(ids: list[int], trainable: bool) -> None:
        if ids:
            segments.append((ids, trainable))

    if "messages" in sample:
        for msg in sample["messages"]:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if role == "system":
                add([SYSTEM_ID] + _encode(sp, content) + [END_ID], False)
            elif role == "user":
                add([USER_ID] + _encode(sp, content) + [END_ID], False)
            elif role == "tool":
                # environment output: context for the model, never a target
                add([TOOL_RESULT_ID] + _encode(sp, content) + [TOOL_END_ID], False)
            elif role == "assistant":
                # the opening tag is a prompt cue, the content is the target
                add([ASSISTANT_ID], False)
                think = (msg.get("thinking") or "").strip()
                if think:
                    add(
                        [THINK_ID] + _encode(sp, think) + [THINK_END_ID],
                        train_on_reasoning,
                    )
                if msg.get("tool_call"):
                    payload = json.dumps(msg["tool_call"], ensure_ascii=False)
                    add([TOOL_CALL_ID] + _encode(sp, payload) + [TOOL_END_ID], True)
                if content:
                    add(_encode(sp, content), True)
                add([END_ID], True)   # teach the model where to stop
    else:
        instruction = (sample.get("instruction") or "").strip()
        extra = (sample.get("input") or "").strip()
        output = (sample.get("output") or "").strip()
        if not instruction or not output:
            return None

        user_text = f"{instruction}\n\n{extra}" if extra else instruction
        add([USER_ID] + _encode(sp, user_text) + [END_ID], False)
        add([ASSISTANT_ID], False)
        add(_encode(sp, output), True)
        add([END_ID], True)

    input_ids: list[int] = [BOS_ID]
    labels: list[int] = [IGNORE_INDEX]
    for ids, trainable in segments:
        input_ids.extend(ids)
        labels.extend(ids if trainable else [IGNORE_INDEX] * len(ids))

    input_ids.append(EOS_ID)
    labels.append(EOS_ID)

    # Truncate from the left of the *prompt* would be better for long chats, but
    # a hard right truncation is predictable; drop examples that lose their target.
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    if all(l == IGNORE_INDEX for l in labels):
        return None

    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    """Instruction-tuning dataset over a JSONL file."""

    def __init__(
        self,
        jsonl_path: str | Path,
        sp,
        max_seq_len: int = 2048,
        train_on_reasoning: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"SFT file not found: {self.path}")

        self.examples: list[dict[str, list[int]]] = []
        skipped = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                built = build_sft_example(
                    sample, sp, max_seq_len, train_on_reasoning
                )
                if built is None:
                    skipped += 1
                    continue
                self.examples.append(built)
                if limit is not None and len(self.examples) >= limit:
                    break

        self.skipped = skipped
        if not self.examples:
            raise ValueError(f"No usable examples parsed from {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def sft_collate(batch: list[dict[str, list[int]]], pad_id: int = 0) -> dict[str, Tensor]:
    """
    Right-pad a batch and emit the key-padding mask.

    Padding positions get IGNORE_INDEX labels so they contribute no loss, and
    the attention mask keeps them out of every attention score. Both are
    required: masking only the loss still lets padding pollute the attention
    softmax for real tokens.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attn = [], [], []

    for ex in batch:
        ids, lab = ex["input_ids"], ex["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [IGNORE_INDEX] * pad)
        attn.append([1] * len(ids) + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }
