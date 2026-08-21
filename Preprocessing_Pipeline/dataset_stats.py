"""
dataset_stats.py
----------------
Computes comprehensive statistics over the tokenized binary cache produced by
dataset_tokenizer.py.  Reads only the index file for length computations;
never loads the full token stream into RAM.

Metrics reported
────────────────
  - Total / valid / invalid documents
  - Total tokens
  - Per-document: mean, median, min, max, p95, p99
  - Documents exceeding 4096 tokens
  - ASCII histogram of document-length distribution
  - Token-range bucket distribution

Usage:
    python dataset_stats.py \
        --cache-dir ./cache \
        [--context-len 4096] \
        [--histogram-bins 20] \
        [--output stats.json]
"""

import argparse
import json
import logging
import math
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

IDX_ENTRY_SIZE = 8  # uint64


# ---------------------------------------------------------------------------
# Index reader
# ---------------------------------------------------------------------------

def load_doc_lengths(idx_path: Path) -> np.ndarray:
    """
    Read the offset index and return an int64 array of per-document token
    counts.  The index stores byte offsets; since each token is 2 bytes
    (uint16), length_in_tokens = (offset[i+1] - offset[i]) // 2.
    """
    size = idx_path.stat().st_size
    n_entries = size // IDX_ENTRY_SIZE
    if n_entries < 2:
        sys.exit("[ERROR] Index file is too small; dataset may be empty.")

    log.info("Reading index: %d entries", n_entries)
    with idx_path.open("rb") as fh:
        raw = fh.read()

    offsets = np.frombuffer(raw, dtype="<u8")  # uint64 little-endian
    lengths_bytes = np.diff(offsets)  # n_entries - 1 = n_documents
    lengths_tokens = lengths_bytes // 2  # uint16 → token count
    return lengths_tokens.astype(np.int64)


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats(lengths: np.ndarray, context_len: int) -> dict:
    n = len(lengths)
    total_tokens = int(lengths.sum())
    non_empty_mask = lengths > 0
    non_empty = lengths[non_empty_mask]
    empty_count = int((lengths == 0).sum())

    if len(non_empty) == 0:
        sys.exit("[ERROR] All documents are empty. Nothing to analyze.")

    stats = {
        "total_documents": n,
        "valid_documents": int(non_empty_mask.sum()),
        "empty_documents": empty_count,
        "total_tokens": total_tokens,
        "mean_tokens": float(np.mean(non_empty)),
        "median_tokens": float(np.median(non_empty)),
        "std_tokens": float(np.std(non_empty)),
        "min_tokens": int(non_empty.min()),
        "max_tokens": int(non_empty.max()),
        "p25_tokens": float(np.percentile(non_empty, 25)),
        "p75_tokens": float(np.percentile(non_empty, 75)),
        "p95_tokens": float(np.percentile(non_empty, 95)),
        "p99_tokens": float(np.percentile(non_empty, 99)),
        "docs_exceeding_context": int((non_empty > context_len).sum()),
        "pct_exceeding_context": float((non_empty > context_len).mean() * 100),
    }
    return stats


def build_histogram(
    lengths: np.ndarray, bins: int = 20
) -> list[dict]:
    non_empty = lengths[lengths > 0]
    lo = int(non_empty.min())
    hi = int(non_empty.max())
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(non_empty, bins=edges)
    buckets = []
    for i in range(bins):
        buckets.append(
            {
                "lo": int(edges[i]),
                "hi": int(edges[i + 1]),
                "count": int(counts[i]),
                "pct": float(counts[i] / len(non_empty) * 100),
            }
        )
    return buckets


def build_range_buckets(lengths: np.ndarray) -> list[dict]:
    """Fixed token-range buckets relevant for LLM training."""
    non_empty = lengths[lengths > 0]
    boundaries = [1, 128, 256, 512, 1024, 2048, 4096, 8192, float("inf")]
    labels = [
        "1–127", "128–255", "256–511", "512–1023",
        "1024–2047", "2048–4095", "4096–8191", "8192+",
    ]
    buckets = []
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (non_empty >= lo) & (non_empty < hi)
        cnt = int(mask.sum())
        buckets.append(
            {
                "range": labels[i],
                "count": cnt,
                "pct": float(cnt / len(non_empty) * 100),
            }
        )
    return buckets


# ---------------------------------------------------------------------------
# ASCII histogram renderer
# ---------------------------------------------------------------------------

def render_histogram(buckets: list[dict], width: int = 50) -> str:
    max_count = max(b["count"] for b in buckets)
    lines = []
    for b in buckets:
        bar_len = int(b["count"] / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(
            f"  {b['lo']:>7} – {b['hi']:>7}  │ {bar:<{width}} {b['count']:>10,}  ({b['pct']:5.1f}%)"
        )
    return "\n".join(lines)


def render_range_buckets(buckets: list[dict], width: int = 40) -> str:
    max_count = max(b["count"] for b in buckets)
    lines = []
    for b in buckets:
        bar_len = int(b["count"] / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(
            f"  {b['range']:>12}  │ {bar:<{width}} {b['count']:>10,}  ({b['pct']:5.1f}%)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def print_report(
    stats: dict,
    hist_buckets: list[dict],
    range_buckets: list[dict],
    context_len: int,
) -> None:
    print(_sep("═"))
    print("  DATASET STATISTICS REPORT")
    print(_sep("═"))

    print("\n[1] DOCUMENT COUNTS")
    print(_sep())
    print(f"  Total documents          : {stats['total_documents']:>14,}")
    print(f"  Valid (non-empty)        : {stats['valid_documents']:>14,}")
    print(f"  Empty documents          : {stats['empty_documents']:>14,}")

    print("\n[2] TOKEN STATISTICS")
    print(_sep())
    print(f"  Total tokens             : {stats['total_tokens']:>14,}  ({stats['total_tokens']/1e9:.3f} B)")
    print(f"  Mean   tokens/doc        : {stats['mean_tokens']:>14.1f}")
    print(f"  Median tokens/doc        : {stats['median_tokens']:>14.1f}")
    print(f"  Std dev                  : {stats['std_tokens']:>14.1f}")
    print(f"  Min    tokens/doc        : {stats['min_tokens']:>14,}")
    print(f"  Max    tokens/doc        : {stats['max_tokens']:>14,}")
    print(f"  P25    tokens/doc        : {stats['p25_tokens']:>14.1f}")
    print(f"  P75    tokens/doc        : {stats['p75_tokens']:>14.1f}")
    print(f"  P95    tokens/doc        : {stats['p95_tokens']:>14.1f}")
    print(f"  P99    tokens/doc        : {stats['p99_tokens']:>14.1f}")

    print(f"\n[3] CONTEXT LENGTH ANALYSIS  (context_len = {context_len})")
    print(_sep())
    print(
        f"  Docs exceeding {context_len:,} tokens  : "
        f"{stats['docs_exceeding_context']:>10,}  "
        f"({stats['pct_exceeding_context']:.2f}%)"
    )
    within = stats['valid_documents'] - stats['docs_exceeding_context']
    print(
        f"  Docs within   {context_len:,} tokens  : "
        f"{within:>10,}  "
        f"({within / stats['valid_documents'] * 100:.2f}%)"
    )

    print("\n[4] DOCUMENT LENGTH HISTOGRAM")
    print(_sep())
    print(f"  {'lo':>7}   {'hi':>7}  │ Distribution")
    print(render_histogram(hist_buckets))

    print("\n[5] TOKEN RANGE BUCKETS")
    print(_sep())
    print(render_range_buckets(range_buckets))

    print()
    print(_sep("═"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute statistics for the tokenized binary cache."
    )
    parser.add_argument(
        "--cache-dir", default="./cache", help="Directory containing tokens.idx"
    )
    parser.add_argument(
        "--context-len", type=int, default=4096, help="Context window length"
    )
    parser.add_argument(
        "--histogram-bins", type=int, default=20, help="Number of histogram bins"
    )
    parser.add_argument(
        "--output", default=None, help="Optional JSON file to write stats to"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    idx_path = cache_dir / "tokens.idx"
    tokens_path = cache_dir / "tokens.bin"

    for p in (idx_path, tokens_path):
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}\n"
                     "        Run dataset_tokenizer.py first.")

    t0 = time.perf_counter()
    lengths = load_doc_lengths(idx_path)
    log.info("Loaded %d document lengths in %.2fs", len(lengths), time.perf_counter() - t0)

    stats = compute_stats(lengths, args.context_len)
    hist_buckets = build_histogram(lengths, bins=args.histogram_bins)
    range_buckets = build_range_buckets(lengths)

    print_report(stats, hist_buckets, range_buckets, args.context_len)

    if args.output:
        out = {
            "stats": stats,
            "histogram": hist_buckets,
            "range_buckets": range_buckets,
        }
        Path(args.output).write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        log.info("Stats written to %s", args.output)


if __name__ == "__main__":
    main()
