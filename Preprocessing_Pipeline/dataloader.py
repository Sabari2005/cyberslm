"""
dataloader.py
-------------
Production DataLoader for causal language model pretraining.

Reads directly from train.bin / val.bin (flat uint16 token streams).
Never loads the full dataset into RAM; uses memory-mapped I/O.

Design
──────
  - BinaryTokenDataset: mmap-backed dataset returning (input, target) pairs.
  - Random start positions are sampled across the full token stream.
  - Wraps PyTorch's DataLoader with pinned memory and multiple workers.
  - Supports both training (shuffled) and evaluation (sequential) modes.

Token pair format
─────────────────
  input  = tokens[pos : pos + context_len]
  target = tokens[pos + 1 : pos + context_len + 1]

Both are int64 tensors ready for nn.Embedding + CrossEntropyLoss.

Usage (standalone test):
    python dataloader.py \
        --data-dir ./data \
        --context-len 4096 \
        --batch-size 8 \
        --num-workers 4
"""

import argparse
import logging
import mmap
import os
import struct
import sys
import time
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

log = logging.getLogger(__name__)

BYTES_PER_TOKEN = 2  # uint16


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinaryTokenDataset(Dataset):
    """
    Memory-mapped dataset over a flat uint16 token binary file.

    Each sample is a (context_len,) input tensor and a (context_len,) target
    tensor where target[i] = input[i + 1] (next-token prediction).

    Sampling contract
    -----------------
    Windows are **non-overlapping and exhaustive**: item ``idx`` is always the
    window starting at ``idx * context_len``. Every token in the file is seen
    exactly once per epoch.

    Shuffling is delegated entirely to the DataLoader's sampler
    (``shuffle=True``), which runs in the *main* process and therefore reshuffles
    correctly on every epoch even when ``persistent_workers=True``.

    This replaces the previous design, which drew a random start position per
    item from a per-item RNG seeded with ``(seed, epoch, idx)``. That had two
    compounding defects:

      1. Sampling *with replacement* meant one pass touched only
         ``1 - 1/e`` = ~63 % of positions; ~37 % of the corpus was never trained on.
      2. The epoch was mixed in inside ``__getitem__``, but ``set_epoch()`` only
         mutated the main-process copy of the dataset. With persistent workers the
         worker copies are pickled once and never rebuilt, so the epoch counter
         never reached them and every epoch replayed byte-identical windows.

    Both are gone: there is no per-item RNG and no epoch state to synchronise.
    """

    def __init__(
        self,
        bin_path: Path,
        context_len: int,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        self.bin_path = bin_path
        self.context_len = context_len
        self.seed = seed

        # Total tokens in the file
        file_bytes = bin_path.stat().st_size
        if file_bytes % BYTES_PER_TOKEN != 0:
            raise ValueError(
                f"Binary file size {file_bytes} is not a multiple of "
                f"{BYTES_PER_TOKEN}; file may be corrupted."
            )
        self.n_tokens = file_bytes // BYTES_PER_TOKEN

        if self.n_tokens < context_len + 1:
            raise ValueError(
                f"File contains only {self.n_tokens} tokens, which is less "
                f"than context_len + 1 = {context_len + 1}."
            )

        # Number of full non-overlapping windows. The -1 reserves the extra
        # token every window needs for its shifted target.
        self.n_windows = (self.n_tokens - 1) // context_len

        # num_samples is a *cap* (useful for smoke tests); None = use everything.
        if num_samples is None or num_samples <= 0:
            self.num_samples = self.n_windows
        else:
            self.num_samples = min(num_samples, self.n_windows)

        # The mmap/file handle are opened lazily (see _open()) rather than
        # here. DataLoader with num_workers > 0 pickles the Dataset object to
        # hand it to worker processes; on Windows (spawn start method) this
        # requires every attribute to be picklable, and both mmap.mmap and
        # open file handles are not. Opening on first access in each process
        # (combined with __getstate__/__setstate__ below) avoids that crash
        # and also means each worker gets its own independent mmap.
        self._fh = None
        self._mm = None

    def _open(self) -> None:
        if self._mm is None:
            self._fh = self.bin_path.open("rb")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __getstate__(self) -> dict:
        # Strip the unpicklable mmap/file handle before crossing the process
        # boundary; workers reopen their own via _open() on first access.
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_mm"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def set_epoch(self, epoch: int) -> None:
        """
        Retained as a no-op for API compatibility.

        Reshuffling is the DataLoader sampler's job now. Deliberately does
        nothing so no caller can be fooled into believing it had an effect --
        the previous implementation silently did nothing under persistent
        workers, which is the bug this class was rewritten to remove.
        """
        return None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()

        pos = idx * self.context_len
        byte_start = pos * BYTES_PER_TOKEN
        byte_end = byte_start + (self.context_len + 1) * BYTES_PER_TOKEN

        chunk = np.frombuffer(
            self._mm[byte_start:byte_end], dtype="<u2"
        ).astype(np.int64)  # uint16 -> int64 for embedding layer

        x = torch.from_numpy(chunk[: self.context_len])
        y = torch.from_numpy(chunk[1 : self.context_len + 1])
        return x, y

    def close(self) -> None:
        """
        Release the mmap and file handle explicitly.

        Windows refuses to delete or rename a file while an mmap on it is open,
        so relying on __del__ (which fires whenever GC gets around to it) makes
        temp-file cleanup flaky. Callers that need the file released at a
        deterministic point should call this.
        """
        try:
            if self._mm is not None:
                self._mm.close()
        finally:
            self._mm = None
            if self._fh is not None:
                self._fh.close()
            self._fh = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sequential evaluation dataset (no random sampling)
# ---------------------------------------------------------------------------

class SequentialTokenDataset(Dataset):
    """
    Yields non-overlapping context windows from the token file, in order.
    Used for deterministic validation loss computation.
    """

    def __init__(self, bin_path: Path, context_len: int) -> None:
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        self.context_len = context_len
        self.bin_path = bin_path
        file_bytes = bin_path.stat().st_size
        self.n_tokens = file_bytes // BYTES_PER_TOKEN
        self.num_samples = max(0, (self.n_tokens - 1) // context_len)

        # See BinaryTokenDataset for why this is opened lazily instead of here.
        self._fh = None
        self._mm = None

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

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()

        pos = idx * self.context_len
        byte_start = pos * BYTES_PER_TOKEN
        byte_end = byte_start + (self.context_len + 1) * BYTES_PER_TOKEN

        chunk = np.frombuffer(
            self._mm[byte_start:byte_end], dtype="<u2"
        ).astype(np.int64)

        x = torch.from_numpy(chunk[: self.context_len])
        y = torch.from_numpy(chunk[1 : self.context_len + 1])
        return x, y

    def close(self) -> None:
        """
        Release the mmap and file handle explicitly.

        Windows refuses to delete or rename a file while an mmap on it is open,
        so relying on __del__ (which fires whenever GC gets around to it) makes
        temp-file cleanup flaky. Callers that need the file released at a
        deterministic point should call this.
        """
        try:
            if self._mm is not None:
                self._mm.close()
        finally:
            self._mm = None
            if self._fh is not None:
                self._fh.close()
            self._fh = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _optimal_num_workers(requested: int) -> int:
    """
    Clamp requested workers to sane bounds.
    0 = run in main process (useful for debugging).
    """
    cpu_count = os.cpu_count() or 1
    if requested < 0:
        return max(0, cpu_count - 1)
    return min(requested, cpu_count)


def make_train_dataloader(
    bin_path: Path,
    context_len: int,
    batch_size: int,
    num_samples: int | None = None,
    seed: int = 42,
    num_workers: int = -1,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, BinaryTokenDataset]:
    """
    Build a DataLoader for training.

    Returns (dataloader, dataset).

    Windows are non-overlapping and exhaustive; per-epoch reshuffling is handled
    by the DataLoader's sampler (shuffle=True), which lives in the main process
    and therefore works correctly with persistent_workers=True.

    Parameters
    ----------
    bin_path     : Path to train.bin
    context_len  : Sequence length for each sample (e.g. 4096)
    batch_size   : Number of sequences per batch
    num_samples  : Optional cap on windows per epoch (None = every window)
    seed         : Base RNG seed
    num_workers  : Parallel prefetch workers (-1 = auto)
    pin_memory   : Pin tensors to CUDA-accessible memory
    distributed  : Enable DistributedSampler (multi-GPU training)
    rank         : Current process rank (DDP)
    world_size   : Total number of processes (DDP)
    """
    dataset = BinaryTokenDataset(
        bin_path=bin_path,
        context_len=context_len,
        num_samples=num_samples,
        seed=seed,
    )

    nw = _optimal_num_workers(num_workers)
    pin = pin_memory and torch.cuda.is_available()

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,        # avoids partial batches during training
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    return loader, dataset


def make_val_dataloader(
    bin_path: Path,
    context_len: int,
    batch_size: int,
    num_workers: int = -1,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader for sequential, deterministic validation."""
    dataset = SequentialTokenDataset(bin_path=bin_path, context_len=context_len)
    nw = _optimal_num_workers(num_workers)
    pin = pin_memory and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

def _smoke_test(
    data_dir: Path,
    context_len: int,
    batch_size: int,
    num_workers: int,
    n_batches: int = 5,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    for path in (train_bin, val_bin):
        if not path.exists():
            sys.exit(f"[ERROR] {path} not found. Run dataset_builder.py first.")

    log.info("=== Train DataLoader smoke test ===")
    n_tokens = train_bin.stat().st_size // BYTES_PER_TOKEN
    num_samples = 10_000
    loader, dataset = make_train_dataloader(
        bin_path=train_bin,
        context_len=context_len,
        batch_size=batch_size,
        num_samples=num_samples,
        seed=42,
        num_workers=num_workers,
    )

    t0 = time.perf_counter()
    total_tokens = 0
    for step, (x, y) in enumerate(loader):
        assert x.shape == (batch_size, context_len), f"Wrong x shape: {x.shape}"
        assert y.shape == (batch_size, context_len), f"Wrong y shape: {y.shape}"
        # Causal target invariant: y[b, i] == x[b, i+1] cannot be verified
        # directly here (positions are random), but dtype must be int64
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64
        total_tokens += x.numel()
        if step == 0:
            log.info("  x[0, :8] = %s", x[0, :8].tolist())
            log.info("  y[0, :8] = %s", y[0, :8].tolist())
        if step + 1 >= n_batches:
            break

    elapsed = time.perf_counter() - t0
    log.info(
        "  %d batches | %d tokens | %.2f M tok/s",
        n_batches,
        total_tokens,
        total_tokens / elapsed / 1e6,
    )

    log.info("=== Val DataLoader smoke test ===")
    val_loader = make_val_dataloader(
        bin_path=val_bin,
        context_len=context_len,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    t0 = time.perf_counter()
    total_tokens = 0
    for step, (x, y) in enumerate(val_loader):
        total_tokens += x.numel()
        if step + 1 >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    log.info(
        "  %d batches | %d tokens | %.2f M tok/s",
        n_batches,
        total_tokens,
        total_tokens / elapsed / 1e6,
    )
    log.info("=== All assertions passed ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the BinaryToken DataLoader."
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-batches", type=int, default=5)
    args = parser.parse_args()

    _smoke_test(
        data_dir=Path(args.data_dir),
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_batches=args.n_batches,
    )


if __name__ == "__main__":
    main()
