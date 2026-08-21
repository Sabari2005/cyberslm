"""Streaming extraction of training text from a JSONL corpus.

Reads a JSONL file where each line is `{"text": "..."}` and writes the
raw text value (one document per line) to disk for consumption by the
SentencePiece trainer. Parsing is distributed across worker processes;
the corpus is never fully materialized in memory.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# How often (in lines) to emit a progress log line.
_PROGRESS_INTERVAL = 1_000_000

# Number of lines handed to a worker process per IPC round-trip. Batching
# amortizes inter-process communication overhead across many small JSON
# records instead of paying it per line.
_CHUNKSIZE = 2000

_ExtractResult = Tuple[Optional[str], Optional[str]]


def _extract_line(raw_line: bytes) -> _ExtractResult:
    """Parse and extract a single JSONL line.

    Returns:
        (text, None)      -- a valid document.
        (None, category)  -- the line was rejected; category names why.
        (None, None)      -- the line was blank; not an error, nothing to do.
    """
    try:
        decoded = raw_line.decode("utf-8")
    except UnicodeDecodeError:
        return None, "invalid_unicode"

    if not decoded.strip():
        return None, None

    try:
        record = json.loads(decoded)
    except json.JSONDecodeError:
        return None, "invalid_json"

    if not isinstance(record, dict) or "text" not in record:
        return None, "missing_text_field"

    text = record["text"]
    if not isinstance(text, str):
        return None, "invalid_text_type"

    return text, None


def _consume(
    results: Iterable[_ExtractResult],
    fout,
    fin,
    total_size: int,
    start_time: float,
) -> Counter:
    """Write cleaned documents to `fout` while tallying statistics."""
    stats: Counter = Counter()

    for text, error in results:
        stats["total_lines"] += 1

        if error is not None:
            stats[error] += 1
        elif text is not None:
            fout.write(text)
            fout.write("\n")
            stats["valid"] += 1
        # else: blank line, silently skipped, not counted as an error.

        if stats["total_lines"] % _PROGRESS_INTERVAL == 0:
            pct = (fin.tell() / total_size * 100) if total_size else 100.0
            elapsed = max(time.monotonic() - start_time, 1e-9)
            rate = stats["total_lines"] / elapsed
            logger.info(
                "Processed %d lines (%.1f%% of file, %.0f lines/sec, %d valid so far)",
                stats["total_lines"], pct, rate, stats["valid"],
            )

    return stats


def extract_corpus(input_path: Path, output_path: Path, num_workers: int) -> Counter:
    """Stream `input_path`, extract the `text` field of every JSONL record,
    and write one document per line to `output_path`.

    Line order is preserved (both across single- and multi-process paths),
    so the same input file always produces a byte-identical output corpus
    regardless of `num_workers`.

    Raises FileNotFoundError, IsADirectoryError, or PermissionError if
    `input_path` cannot be read, or PermissionError if `output_path`
    cannot be written. These are left uncaught so the caller can present
    a tailored message.

    Returns a Counter with keys: total_lines, valid, invalid_json,
    missing_text_field, invalid_text_type, invalid_unicode.
    """
    total_size = input_path.stat().st_size
    start_time = time.monotonic()
    num_workers = max(1, num_workers)

    with open(input_path, "rb") as fin, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        if num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                results: Iterator[_ExtractResult] = pool.imap(
                    _extract_line, fin, chunksize=_CHUNKSIZE
                )
                stats = _consume(results, fout, fin, total_size, start_time)
        else:
            results = (_extract_line(line) for line in fin)
            stats = _consume(results, fout, fin, total_size, start_time)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Extraction complete in %.1fs: %d lines read, %d valid documents written",
        elapsed, stats["total_lines"], stats["valid"],
    )
    return stats
