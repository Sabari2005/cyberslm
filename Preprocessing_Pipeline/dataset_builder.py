"""
dataset_builder.py
------------------
Reads the tokenized binary cache, performs a deterministic 95/5 train/val
split, and writes flat uint16 binary files optimised for sequential reads
during training.

Output files
────────────
  train.bin   – flat uint16 token stream (little-endian)
  val.bin     – flat uint16 token stream (little-endian)
  split.json  – document indices for each split (for reproducibility)

Design notes
────────────
- The split is done at the DOCUMENT level, so context windows never straddle
  a train/val boundary.
- Documents are shuffled with a fixed seed before assignment, giving an
  unbiased but deterministic split.
- Tokens are written sequentially without gaps; downstream DataLoader uses
  context-length windows, not document boundaries.
- The copy is streamed in fixed-size chunks; RAM usage is O(chunk_size).

Usage:
    python dataset_builder.py \
        --cache-dir ./cache \
        --output-dir ./data \
        [--val-ratio 0.05] \
        [--seed 42] \
        [--chunk-size 67108864]
"""

import argparse
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IDX_ENTRY_SIZE = 8   # uint64
BYTES_PER_TOKEN = 2  # uint16


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_offsets(idx_path: Path) -> np.ndarray:
    """Return the raw uint64 byte-offset array."""
    with idx_path.open("rb") as fh:
        raw = fh.read()
    return np.frombuffer(raw, dtype="<u8").copy()


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------

def split_document_indices(
    n_docs: int,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle document indices with the given seed and split 95/5.
    Returns (train_indices, val_indices), both sorted for sequential access.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_docs, dtype=np.int64)
    rng.shuffle(indices)
    n_val = max(1, int(n_docs * val_ratio))
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Sequential binary copy
# ---------------------------------------------------------------------------

def _copy_documents(
    src: Path,
    offsets: np.ndarray,
    doc_indices: np.ndarray,
    dst: Path,
    chunk_bytes: int,
    label: str,
) -> int:
    """
    Copy token bytes for doc_indices (sorted) from src to dst.
    Returns total tokens written.
    """
    total_tokens = 0
    total_bytes = 0

    # Compute total bytes needed for progress reporting
    for i in doc_indices:
        total_bytes += int(offsets[i + 1] - offsets[i])

    written_bytes = 0
    t0 = time.perf_counter()

    with src.open("rb") as sf, dst.open("wb") as df:
        # Buffer writes for efficient I/O
        buf = bytearray()
        flush_at = chunk_bytes

        for i in doc_indices:
            start = int(offsets[i])
            end = int(offsets[i + 1])
            length = end - start
            if length == 0:
                continue
            sf.seek(start)
            data = sf.read(length)
            buf.extend(data)
            written_bytes += length
            total_tokens += length // BYTES_PER_TOKEN

            if len(buf) >= flush_at:
                df.write(buf)
                buf.clear()
                elapsed = time.perf_counter() - t0
                throughput = written_bytes / elapsed / 1e9 if elapsed > 0 else 0
                pct = written_bytes / total_bytes * 100 if total_bytes > 0 else 0
                log.info(
                    "%s: %5.1f%%  %6.2f GB  %.2f GB/s",
                    label,
                    pct,
                    written_bytes / 1e9,
                    throughput,
                )

        if buf:
            df.write(buf)

    return total_tokens


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(
    cache_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    chunk_bytes: int,
) -> None:
    tokens_path = cache_dir / "tokens.bin"
    idx_path = cache_dir / "tokens.idx"

    for p in (tokens_path, idx_path):
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    split_meta_path = output_dir / "split.json"

    log.info("Loading offset index …")
    offsets = load_offsets(idx_path)
    n_docs = len(offsets) - 1  # sentinel entry at end
    log.info("Documents in cache: %d", n_docs)

    if n_docs == 0:
        sys.exit("[ERROR] No documents found in index.")

    log.info("Splitting: val_ratio=%.3f  seed=%d", val_ratio, seed)
    train_idx, val_idx = split_document_indices(n_docs, val_ratio, seed)
    log.info(
        "Split: train=%d docs  val=%d docs",
        len(train_idx),
        len(val_idx),
    )

    log.info("Writing train.bin …")
    train_tokens = _copy_documents(
        tokens_path, offsets, train_idx, train_path, chunk_bytes, "train"
    )
    log.info("train.bin: %d tokens  (%.3f B)  %.3f GB",
             train_tokens, train_tokens / 1e9,
             train_path.stat().st_size / 1e9)

    log.info("Writing val.bin …")
    val_tokens = _copy_documents(
        tokens_path, offsets, val_idx, val_path, chunk_bytes, "val  "
    )
    log.info("val.bin: %d tokens  (%.3f B)  %.3f GB",
             val_tokens, val_tokens / 1e9,
             val_path.stat().st_size / 1e9)

    # Persist split metadata for reproducibility
    meta = {
        "seed": seed,
        "val_ratio": val_ratio,
        "n_docs_total": n_docs,
        "n_docs_train": len(train_idx),
        "n_docs_val": len(val_idx),
        "n_tokens_train": train_tokens,
        "n_tokens_val": val_tokens,
        "train_doc_indices": train_idx.tolist(),
        "val_doc_indices": val_idx.tolist(),
    }
    split_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total_tokens = train_tokens + val_tokens
    actual_val_ratio = val_tokens / total_tokens if total_tokens > 0 else 0.0

    log.info("─" * 60)
    log.info("train.bin     : %d tokens  (%.3f B)", train_tokens, train_tokens / 1e9)
    log.info("val.bin       : %d tokens  (%.3f B)", val_tokens, val_tokens / 1e9)
    log.info("Actual val %%  : %.3f%%", actual_val_ratio * 100)
    log.info("split.json    : %s", split_meta_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/val binary files from the tokenized cache."
    )
    parser.add_argument(
        "--cache-dir", default="./cache", help="Directory with tokens.bin / tokens.idx"
    )
    parser.add_argument(
        "--output-dir", default="./data", help="Destination for train.bin and val.bin"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of documents for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic split"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64 * 1024 * 1024,
        help="Write-buffer size in bytes (default: 64 MB)",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        sys.exit("[ERROR] --val-ratio must be in (0, 1).")

    build_dataset(
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        chunk_bytes=args.chunk_size,
    )


if __name__ == "__main__":
    main()
