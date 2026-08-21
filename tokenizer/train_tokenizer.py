#!/usr/bin/env python3
"""Train a SentencePiece BPE tokenizer for a cybersecurity-focused SLM.

Reads a JSONL corpus (one {"text": ...} object per line), extracts the
text field without additional cleaning, and trains a SentencePiece BPE tokenizer suitable for
pretraining, instruction tuning, and tool-calling on a decoder-only
transformer. See README.md for configuration details and rationale.

Usage:
    python train_tokenizer.py --input corpus.jsonl --output-dir ./tokenizer_output
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import sentencepiece as spm

from preprocessing import extract_corpus

logger = logging.getLogger("train_tokenizer")

DEFAULT_VOCAB_SIZE = 32000
DEFAULT_CHARACTER_COVERAGE = 0.9995
DEFAULT_INPUT_SENTENCE_SIZE = 20_000_000
DEFAULT_MAX_SENTENCE_LENGTH = 16384
DEFAULT_MODEL_NAME = "tokenizer"

# Representative strings covering every technical category called out in
# the spec. Used only for a post-training sanity check -- never written to
# any output file.
_SMOKE_TEST_SAMPLES = (
    "CVE-2024-3400 affects PAN-OS GlobalProtect gateways running 11.0 and 11.1.",
    "Connect to 192.168.1.1 on port 8443 using TLS 1.3 with a valid certificate.",
    "curl -sSL https://example.com/install.sh | bash -s -- --verbose",
    "SELECT username, password_hash FROM users WHERE last_login > '2024-01-01';",
    "$hash = Get-FileHash -Algorithm SHA256 -Path C:\\Windows\\System32\\cmd.exe",
    "def scan_ports(host: str, ports: list[int]) -> dict:\n    return {}",
    '{"level": "ALERT", "src_ip": "10.0.0.5", "sha256": "e3b0c44298fc1c14"}',
    "/var/log/auth.log",
    "version: '3.8'\nservices:\n  web:\n    image: nginx:latest",
)


def _available_cpus() -> int:
    """Number of CPUs actually schedulable by this process.

    Prefers sched_getaffinity (respects container/cgroup CPU quotas on
    Linux) over os.cpu_count() (which reports host-wide core count and can
    overstate what's usable inside a container).
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the JSONL training corpus.")
    parser.add_argument("--output-dir", default=Path("./tokenizer_output"), type=Path, help="Directory for tokenizer.model / tokenizer.vocab.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Output file prefix (produces <name>.model and <name>.vocab).")
    parser.add_argument("--vocab-size", default=DEFAULT_VOCAB_SIZE, type=int, help="Target vocabulary size.")
    parser.add_argument("--character-coverage", default=DEFAULT_CHARACTER_COVERAGE, type=float, help="Fraction of corpus characters guaranteed a dedicated piece.")
    parser.add_argument("--input-sentence-size", default=DEFAULT_INPUT_SENTENCE_SIZE, type=int, help="Cap on training lines sampled from the corpus (0 = use every line).")
    parser.add_argument("--max-sentence-length", default=DEFAULT_MAX_SENTENCE_LENGTH, type=int, help="Max bytes per training line; longer lines are skipped by the trainer.")
    parser.add_argument("--num-threads", default=_available_cpus(), type=int, help="Threads used internally by the SentencePiece trainer.")
    parser.add_argument("--workers", default=_available_cpus(), type=int, help="Worker processes used for JSONL preprocessing.")
    parser.add_argument("--keep-corpus", action="store_true", help="Keep the intermediate extracted-text corpus file instead of deleting it after training.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def _log_extraction_stats(stats) -> None:
    total = stats["total_lines"]
    valid = stats["valid"]
    logger.info("Extraction summary: %d lines read, %d documents written (%.2f%%)",
                total, valid, (valid / total * 100) if total else 0.0)
    for category in ("invalid_json", "missing_text_field", "invalid_text_type",
                      "invalid_unicode"):
        count = stats.get(category, 0)
        if count:
            logger.warning("  skipped %d lines: %s", count, category)


def train_sentencepiece(corpus_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    """Invoke the SentencePiece trainer. Returns the path to the .model file."""
    model_prefix = str(output_dir / args.model_name)
    logger.info(
        "Starting SentencePiece BPE training: vocab_size=%d, threads=%d",
        args.vocab_size, args.num_threads,
    )

    spm.SentencePieceTrainer.train(
        # --- corpus ---
        input=str(corpus_path),
        input_format="text",
        model_prefix=model_prefix,
        # --- algorithm ---
        model_type="bpe",
        vocab_size=args.vocab_size,
        hard_vocab_limit=True,
        # --- normalization: preserve case, exact whitespace, digits ---
        normalization_rule_name="nfkc",
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        add_dummy_prefix=True,
        split_digits=False,
        split_by_number=False,
        character_coverage=args.character_coverage,
        byte_fallback=True,
        max_sentencepiece_length=16,
        # --- corpus scaling ---
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=args.max_sentence_length,
        train_extremely_large_corpus=True,
        num_threads=args.num_threads,
        # --- special tokens ---
        pad_id=0, pad_piece="<pad>",
        unk_id=1, unk_piece="<unk>",
        bos_id=2, bos_piece="<bos>",
        eos_id=3, eos_piece="<eos>",
    )

    logger.info("Training complete: %s.model, %s.vocab", model_prefix, model_prefix)
    return Path(f"{model_prefix}.model")


def run_smoke_tests(model_path: Path) -> None:
    """Load the trained model and sanity-check it on representative
    cybersecurity/code strings. Raises RuntimeError if the model cannot
    be loaded or a basic encode/decode round trip fails.
    """
    logger.info("Running post-training validation on %s", model_path)
    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    logger.info("  vocab_size=%d  pad=%d(%s) unk=%d(%s) bos=%d(%s) eos=%d(%s)",
                sp.vocab_size(),
                sp.pad_id(), sp.id_to_piece(sp.pad_id()),
                sp.unk_id(), sp.id_to_piece(sp.unk_id()),
                sp.bos_id(), sp.id_to_piece(sp.bos_id()),
                sp.eos_id(), sp.id_to_piece(sp.eos_id()))

    failures = 0
    for text in _SMOKE_TEST_SAMPLES:
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        ok = decoded == text
        failures += not ok
        logger.info("  [%s] %d tokens :: %r", "OK" if ok else "MISMATCH", len(ids), text[:60])

    if failures:
        raise RuntimeError(
            f"{failures} smoke-test sample(s) failed to round-trip through "
            "encode/decode. The model file was written but may be misconfigured."
        )
    logger.info("All smoke tests passed.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-8s %(message)s")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create output directory %s: %s", args.output_dir, e)
        return 1

    corpus_path = args.output_dir / f".{args.model_name}_corpus.txt"

    try:
        stats = extract_corpus(args.input, corpus_path, max(1, args.workers))
    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input)
        return 1
    except IsADirectoryError:
        logger.error("Input path is a directory, not a file: %s", args.input)
        return 1
    except PermissionError:
        logger.error("Permission denied reading input file: %s", args.input)
        return 1
    except KeyboardInterrupt:
        logger.warning("Preprocessing interrupted by user. Cleaning up partial output.")
        corpus_path.unlink(missing_ok=True)
        return 130

    _log_extraction_stats(stats)

    if stats["valid"] == 0:
        logger.error("No training documents extracted from %s. Aborting.", args.input)
        corpus_path.unlink(missing_ok=True)
        return 1

    try:
        model_path = train_sentencepiece(corpus_path, args.output_dir, args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Cleaning up partial output.")
        for ext in (".model", ".vocab"):
            (args.output_dir / f"{args.model_name}{ext}").unlink(missing_ok=True)
        return 130
    except RuntimeError as e:
        logger.error("SentencePiece training failed: %s", e)
        logger.error("Common causes: --vocab-size too large for the amount of "
                     "training text (too few extracted documents -- see the "
                     "extraction summary above), or --vocab-size too small to "
                     "fit the special tokens, the 256 byte-fallback tokens, and "
                     "the corpus alphabet (roughly 260+ is the practical floor "
                     "with byte_fallback enabled).")
        return 1
    finally:
        if not args.keep_corpus:
            corpus_path.unlink(missing_ok=True)

    try:
        run_smoke_tests(model_path)
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    logger.info("Done. Tokenizer written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
