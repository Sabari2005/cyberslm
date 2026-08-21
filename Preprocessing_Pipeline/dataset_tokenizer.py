"""
dataset_tokenizer.py
--------------------
Streams a JSONL dataset, encodes each document with SentencePiece BPE, and
writes token IDs to a binary cache file alongside a companion offset index.

Output layout
─────────────
  tokens.bin    – flat sequence of uint16 token IDs (little-endian)
  tokens.idx    – per-document byte offsets into tokens.bin (uint64, LE)
                  Entry i: byte offset of document i's first token.
                  A sentinel final entry holds the total byte length,
                  allowing O(1) document length queries.
  tokenize.log  – human-readable processing summary

The binary cache is an intermediate artifact consumed by dataset_stats.py
and dataset_builder.py.  It is NOT the final train/val split.

Usage:
    python dataset_tokenizer.py \
        --input  dataset.jsonl \
        --model  tokenizer.model \
        --output-dir ./cache \
        [--workers 4] \
        [--batch-size 1024] \
        [--add-bos] \
        [--add-eos]
"""

import argparse
import io
import json
import logging
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path
from typing import Iterator

import sentencepiece as spm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UINT16_MAX = 65535
IDX_ENTRY_SIZE = 8          # uint64
REPORT_INTERVAL = 100_000   # documents between progress logs
READ_BUFFER = 64 * 1024 * 1024  # 64 MB read buffer for JSONL

# ---------------------------------------------------------------------------
# Tokenizer loader (per-process, avoids pickling the sp object)
# ---------------------------------------------------------------------------

_SP: spm.SentencePieceProcessor | None = None
_MODEL_PATH: str = ""
_ADD_BOS: bool = False
_ADD_EOS: bool = False


def _init_worker(model_path: str, add_bos: bool, add_eos: bool) -> None:
    global _SP, _MODEL_PATH, _ADD_BOS, _ADD_EOS
    _MODEL_PATH = model_path
    _ADD_BOS = add_bos
    _ADD_EOS = add_eos
    _SP = spm.SentencePieceProcessor()
    _SP.Load(model_path)


def _encode_batch(texts: list[str]) -> list[list[int]]:
    assert _SP is not None
    results: list[list[int]] = []
    for text in texts:
        ids = _SP.EncodeAsIds(text)
        if _ADD_BOS:
            ids = [_SP.bos_id()] + ids
        if _ADD_EOS:
            ids = ids + [_SP.eos_id()]
        results.append(ids)
    return results


# ---------------------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------------------

def _iter_jsonl(
    path: Path,
) -> Iterator[tuple[int, str | None, str | None]]:
    """
    Yields (line_number, text, error_reason).
    text is None when the line should be skipped.
    """
    with path.open("rb", buffering=READ_BUFFER) as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.rstrip(b"\n\r")
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                yield lineno, None, f"json_error: {exc}"
                continue
            if not isinstance(obj, dict):
                yield lineno, None, "not_a_dict"
                continue
            text = obj.get("text")
            if text is None:
                yield lineno, None, "missing_text_field"
                continue
            if not isinstance(text, str):
                yield lineno, None, f"text_not_string: {type(text).__name__}"
                continue
            text = text.strip()
            if not text:
                yield lineno, None, "empty_text"
                continue
            # Sanitise invalid surrogate code points that crash SentencePiece
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                text = text.encode("utf-8", errors="replace").decode("utf-8")
            yield lineno, text, None


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def _batched(
    iterator: Iterator[tuple[int, str | None, str | None]],
    batch_size: int,
) -> Iterator[tuple[list[tuple[int, str]], dict[str, int]]]:
    """
    Yields (valid_batch, skip_counts) where valid_batch is a list of
    (lineno, text) and skip_counts tallies skipped lines by reason.
    """
    batch: list[tuple[int, str]] = []
    skips: dict[str, int] = {}
    for lineno, text, reason in iterator:
        if text is None:
            skips[reason] = skips.get(reason, 0) + 1
            continue
        batch.append((lineno, text))
        if len(batch) == batch_size:
            yield batch, skips
            batch = []
            skips = {}
    if batch:
        yield batch, skips


# ---------------------------------------------------------------------------
# Binary writer helpers
# ---------------------------------------------------------------------------

def _pack_tokens(ids: list[int]) -> bytes:
    """Pack token IDs as little-endian uint16."""
    return struct.pack(f"<{len(ids)}H", *ids)


# ---------------------------------------------------------------------------
# Main tokenization pass
# ---------------------------------------------------------------------------

def tokenize_dataset(
    input_path: Path,
    output_dir: Path,
    model_path: str,
    add_bos: bool,
    add_eos: bool,
    workers: int,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "tokens.bin"
    idx_path = output_dir / "tokens.idx"
    log_path = output_dir / "tokenize.log"

    for p in (tokens_path, idx_path):
        if p.exists():
            p.unlink()

    log.info("Input   : %s  (%.2f GB)", input_path, input_path.stat().st_size / 1e9)
    log.info("Output  : %s", output_dir)
    log.info("Workers : %d", workers)
    log.info("Batch   : %d docs", batch_size)

    total_docs = 0
    valid_docs = 0
    total_tokens = 0
    skip_totals: dict[str, int] = {}
    t0 = time.perf_counter()

    pool_ctx = mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(model_path, add_bos, add_eos),
    )

    with pool_ctx as pool, \
         tokens_path.open("wb") as tf, \
         idx_path.open("wb") as ixf:

        # Write sentinel: first offset is 0
        byte_offset = 0
        ixf.write(struct.pack("<Q", byte_offset))

        jsonl_iter = _iter_jsonl(input_path)

        for batch, skips in _batched(jsonl_iter, batch_size):
            texts = [t for _, t in batch]
            total_docs += len(texts) + sum(skips.values())

            # Encode in worker pool
            encoded_list: list[list[int]] = pool.apply(
                _encode_batch, (texts,)
            )

            for ids in encoded_list:
                if not ids:
                    skips["empty_after_encode"] = skips.get("empty_after_encode", 0) + 1
                    # Still write sentinel offset (zero-length document)
                    ixf.write(struct.pack("<Q", byte_offset))
                    valid_docs += 1
                    continue
                raw = _pack_tokens(ids)
                tf.write(raw)
                byte_offset += len(raw)
                ixf.write(struct.pack("<Q", byte_offset))
                total_tokens += len(ids)
                valid_docs += 1

            for reason, cnt in skips.items():
                skip_totals[reason] = skip_totals.get(reason, 0) + cnt

            if total_docs % REPORT_INTERVAL < batch_size:
                elapsed = time.perf_counter() - t0
                rate = total_tokens / elapsed if elapsed > 0 else 0
                log.info(
                    "processed %10d docs | %12d tokens | %.2f Mtok/s",
                    total_docs,
                    total_tokens,
                    rate / 1e6,
                )

    elapsed = time.perf_counter() - t0
    skipped = sum(skip_totals.values())

    # Write summary log
    summary_lines = [
        "=== Tokenization Summary ===",
        f"Input file      : {input_path}",
        f"Total lines     : {total_docs:,}",
        f"Valid documents : {valid_docs:,}",
        f"Skipped         : {skipped:,}",
        "",
        "Skip breakdown:",
    ]
    for reason, cnt in sorted(skip_totals.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {reason:<30}: {cnt:,}")
    summary_lines += [
        "",
        f"Total tokens    : {total_tokens:,}",
        f"Elapsed         : {elapsed:.1f}s",
        f"Throughput      : {total_tokens/elapsed/1e6:.2f} Mtok/s" if elapsed > 0 else "",
        f"Tokens file     : {tokens_path}  ({tokens_path.stat().st_size/1e9:.3f} GB)",
        f"Index file      : {idx_path}  ({idx_path.stat().st_size/1e6:.1f} MB)",
    ]
    log_text = "\n".join(summary_lines)
    log_path.write_text(log_text, encoding="utf-8")

    log.info("─" * 60)
    log.info("Total lines     : %12d", total_docs)
    log.info("Valid documents : %12d", valid_docs)
    log.info("Skipped         : %12d", skipped)
    log.info("Total tokens    : %12d  (%.2f B)", total_tokens, total_tokens / 1e9)
    log.info("Elapsed         : %10.1f s", elapsed)
    if elapsed > 0:
        log.info("Throughput      : %10.2f Mtok/s", total_tokens / elapsed / 1e6)
    log.info("tokens.bin      : %s", tokens_path)
    log.info("tokens.idx      : %s", idx_path)
    log.info("tokenize.log    : %s", log_path)

    if skipped > 0:
        log.warning("Skip breakdown:")
        for reason, cnt in sorted(skip_totals.items(), key=lambda x: -x[1]):
            log.warning("  %-30s : %d", reason, cnt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize a JSONL dataset using a SentencePiece model."
    )
    parser.add_argument("--input", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--model", required=True, help="Path to tokenizer.model")
    parser.add_argument(
        "--output-dir", default="./cache", help="Directory for output files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Parallel worker processes for encoding",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Documents per encoding batch",
    )
    parser.add_argument(
        "--add-bos", action="store_true", help="Prepend BOS token to each document"
    )
    parser.add_argument(
        "--add-eos",
        dest="add_eos",
        action="store_true",
        default=True,
        help="Append EOS (id 3) after each document (DEFAULT: on). This is "
             "required for causal-LM pretraining so packed context windows "
             "contain real document boundaries and the model learns to stop.",
    )
    parser.add_argument(
        "--no-add-eos",
        dest="add_eos",
        action="store_false",
        help="Disable the end-of-document EOS separator (not recommended).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {args.input}")

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"[ERROR] Tokenizer model not found: {args.model}")

    # Validate tokenizer loads cleanly before spawning workers
    sp = spm.SentencePieceProcessor()
    try:
        sp.Load(str(model_path))
    except Exception as exc:
        sys.exit(f"[ERROR] Corrupted tokenizer: {exc}")
    log.info("Tokenizer loaded: vocab_size=%d", sp.GetPieceSize())

    tokenize_dataset(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        model_path=str(model_path),
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
