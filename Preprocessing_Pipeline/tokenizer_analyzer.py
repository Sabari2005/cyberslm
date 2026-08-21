"""
tokenizer_analyzer.py
---------------------
Analyzes a trained SentencePiece BPE tokenizer and reports detailed statistics.
Provides encode/decode round-trip verification.

Usage:
    python tokenizer_analyzer.py \
        --model tokenizer.model \
        --vocab tokenizer.vocab \
        [--top-k 50]
"""

import argparse
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    path = Path(model_path)
    if not path.exists():
        sys.exit(f"[ERROR] Tokenizer model not found: {model_path}")
    sp = spm.SentencePieceProcessor()
    try:
        sp.Load(str(path))
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to load tokenizer: {exc}")
    return sp


def load_vocab(vocab_path: str) -> list[tuple[str, float]]:
    path = Path(vocab_path)
    if not path.exists():
        sys.exit(f"[ERROR] Vocabulary file not found: {vocab_path}")
    entries: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            piece = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0
            entries.append((piece, score))
    return entries


def collect_special_tokens(sp: spm.SentencePieceProcessor) -> dict[str, int]:
    specials: dict[str, int] = {}
    candidates = {
        "bos": sp.bos_id(),
        "eos": sp.eos_id(),
        "unk": sp.unk_id(),
        "pad": sp.pad_id(),
    }
    for name, tid in candidates.items():
        if tid != -1:
            specials[name] = tid
    return specials


def analyze_pieces(
    sp: spm.SentencePieceProcessor,
    vocab_entries: list[tuple[str, float]],
    top_k: int,
) -> dict:
    vocab_size = sp.GetPieceSize()
    byte_pieces = 0
    user_defined = 0
    control_pieces = 0
    unknown_pieces = 0
    normal_pieces = 0

    byte_piece_ids: list[int] = []
    lengths: list[int] = []

    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        if sp.IsUnknown(i):
            unknown_pieces += 1
        elif sp.IsControl(i):
            control_pieces += 1
        elif sp.IsByte(i):
            byte_pieces += 1
            byte_piece_ids.append(i)
        else:
            # estimate user-defined vs normal by leading underscore convention
            if piece.startswith("<") and piece.endswith(">"):
                user_defined += 1
            else:
                normal_pieces += 1
            stripped = piece.lstrip("\u2581")  # ▁ is SentencePiece space marker
            lengths.append(len(stripped) if stripped else len(piece))

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    # top-k by vocab score (lower = more frequent in BPE)
    sorted_vocab = sorted(vocab_entries, key=lambda x: x[1], reverse=True)
    top_pieces = sorted_vocab[:top_k]

    # character coverage estimate: unique Unicode characters representable
    covered_chars: set[str] = set()
    for i in range(vocab_size):
        if sp.IsUnknown(i) or sp.IsControl(i) or sp.IsByte(i):
            continue
        piece = sp.IdToPiece(i).lstrip("\u2581")
        for ch in piece:
            covered_chars.add(ch)

    return {
        "vocab_size": vocab_size,
        "normal_pieces": normal_pieces,
        "byte_pieces": byte_pieces,
        "control_pieces": control_pieces,
        "user_defined_pieces": user_defined,
        "unknown_pieces": unknown_pieces,
        "avg_token_length": avg_len,
        "unique_chars_covered": len(covered_chars),
        "top_pieces": top_pieces,
    }


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

VERIFICATION_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "CVE-2024-1234: Buffer overflow in libssl allows remote code execution.",
    "SELECT * FROM users WHERE id = 1 OR '1'='1';",
    "#!/bin/bash\necho 'Hello, World!'\nnetstat -tulnp | grep LISTEN",
    "def exploit(target: str) -> bytes:\n    return b'\\x90' * 100",
    "MITRE ATT&CK T1059.001: PowerShell execution via encoded command.",
    "Привет мир! 你好世界! مرحبا بالعالم",
    "🔒 Encryption: AES-256-GCM with PBKDF2 key derivation.",
]


def verify_roundtrip(sp: spm.SentencePieceProcessor) -> list[dict]:
    results = []
    for sample in VERIFICATION_SAMPLES:
        try:
            ids = sp.EncodeAsIds(sample)
            decoded = sp.DecodeIds(ids)
            match = sample == decoded
            results.append({
                "input": sample,
                "token_count": len(ids),
                "first_tokens": ids[:8],
                "decoded": decoded,
                "match": match,
            })
        except Exception as exc:
            results.append({
                "input": sample,
                "token_count": 0,
                "first_tokens": [],
                "decoded": "",
                "match": False,
                "error": str(exc),
            })
    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def print_report(
    sp: spm.SentencePieceProcessor,
    analysis: dict,
    specials: dict,
    roundtrip: list[dict],
    top_k: int,
) -> None:
    print(_sep("═"))
    print("  TOKENIZER ANALYSIS REPORT")
    print(_sep("═"))

    print("\n[1] VOCABULARY STATISTICS")
    print(_sep())
    vs = analysis["vocab_size"]
    util = (analysis["normal_pieces"] / vs * 100) if vs else 0.0
    print(f"  Vocabulary size          : {vs:>10,}")
    print(f"  Normal (BPE) pieces      : {analysis['normal_pieces']:>10,}  ({analysis['normal_pieces']/vs*100:.2f}%)")
    print(f"  Byte fallback pieces     : {analysis['byte_pieces']:>10,}  ({analysis['byte_pieces']/vs*100:.2f}%)")
    print(f"  Control pieces           : {analysis['control_pieces']:>10,}  ({analysis['control_pieces']/vs*100:.2f}%)")
    print(f"  User-defined pieces      : {analysis['user_defined_pieces']:>10,}  ({analysis['user_defined_pieces']/vs*100:.2f}%)")
    print(f"  Unknown pieces           : {analysis['unknown_pieces']:>10,}")
    print(f"  Vocabulary utilization   : {util:>10.2f}%")
    print(f"  Unique chars covered     : {analysis['unique_chars_covered']:>10,}")
    print(f"  Average token length     : {analysis['avg_token_length']:>10.3f} chars")

    print("\n[2] SPECIAL TOKENS")
    print(_sep())
    if specials:
        for name, tid in sorted(specials.items(), key=lambda x: x[1]):
            piece = sp.IdToPiece(tid)
            print(f"  {name.upper():<6} id={tid:<6}  piece={piece!r}")
    else:
        print("  No standard special tokens detected.")

    print(f"\n[3] TOP {top_k} VOCABULARY PIECES (by score)")
    print(_sep())
    print(f"  {'ID':>6}  {'Score':>10}  Piece")
    print(f"  {'──':>6}  {'─────':>10}  ─────")
    for rank, (piece, score) in enumerate(analysis["top_pieces"][:top_k], 1):
        pid = sp.PieceToId(piece) if sp.PieceToId(piece) != 0 else "?"
        safe_piece = repr(piece) if len(piece) < 20 else repr(piece[:20]) + "…"
        print(f"  {pid:>6}  {score:>10.4f}  {safe_piece}")

    print("\n[4] ENCODE → DECODE ROUND-TRIP VERIFICATION")
    print(_sep())
    all_pass = True
    for i, r in enumerate(roundtrip, 1):
        status = "✓ PASS" if r["match"] else "✗ FAIL"
        if not r["match"]:
            all_pass = False
        preview = r["input"][:60].replace("\n", "↵")
        print(f"  [{i:2}] {status}  tokens={r['token_count']:<5}  {preview!r}")
        if not r["match"]:
            print(f"       INPUT   : {r['input'][:80]!r}")
            print(f"       DECODED : {r['decoded'][:80]!r}")
        if "error" in r:
            print(f"       ERROR   : {r['error']}")
    print()
    if all_pass:
        print("  All round-trip verifications passed.")
    else:
        failed = sum(1 for r in roundtrip if not r["match"])
        print(f"  WARNING: {failed}/{len(roundtrip)} round-trip verifications FAILED.")

    print()
    print(_sep("═"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a trained SentencePiece BPE tokenizer."
    )
    parser.add_argument("--model", required=True, help="Path to tokenizer.model")
    parser.add_argument("--vocab", required=True, help="Path to tokenizer.vocab")
    parser.add_argument(
        "--top-k", type=int, default=50, help="Number of top vocab pieces to display"
    )
    args = parser.parse_args()

    sp = load_tokenizer(args.model)
    vocab_entries = load_vocab(args.vocab)
    specials = collect_special_tokens(sp)
    analysis = analyze_pieces(sp, vocab_entries, args.top_k)
    roundtrip = verify_roundtrip(sp)

    print_report(sp, analysis, specials, roundtrip, args.top_k)


if __name__ == "__main__":
    main()
