"""
Build a HuggingFace fast tokenizer from the CyberSLM SentencePiece BPE model.

    python cyberslm/scripts/build_hf_tokenizer.py --out hf_export/base

Why this exists: transformers' own converters do not handle this file.
`LlamaTokenizer(vocab_file=...)` silently yields vocab_size 3 (LLaMA's converter
assumes a Unigram model; ours is BPE), and `SpmConverter` raises inside
`SpmExtractor.extract` in transformers 5.15.

A tokenizer that silently produces an empty id list would not crash the
benchmark -- it would just score the model at chance. So this builds the BPE
model explicitly and then refuses to write anything unless it reproduces
SentencePiece's output exactly on a corpus of real text.

Merge extraction: SentencePiece stores pieces with scores but not merges. For
each piece, the merge that created it is the split (left, right) where both
halves are in the vocab and score(left) + score(right) is highest. Ordering
merges by the merged piece's score recovers the original merge order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Text the tokenizer must round-trip identically to SentencePiece. Deliberately
# includes the things that break tokenizers: code, hex, CVEs, punctuation runs,
# unicode, and byte-fallback territory.
PROBES = [
    "SQL injection is a common vulnerability.",
    "CVE-2024-3400 affects PAN-OS GlobalProtect gateways running 11.0 and 11.1.",
    "curl -sSL https://example.com/install.sh | bash -s -- --verbose",
    "SELECT username, password_hash FROM users WHERE last_login > '2024-01-01';",
    "$hash = Get-FileHash -Algorithm SHA256 -Path C:\\Windows\\System32\\cmd.exe",
    "def scan_ports(host: str, ports: list[int]) -> dict:\n    return {}",
    '{"level": "ALERT", "src_ip": "10.0.0.5", "sha256": "e3b0c44298fc1c14"}',
    "192.168.1.100:443",
    "The **Hashing** and IKE  \t  multiple   spaces.",
    "naive café résumé — em dash, ellipsis…",
    "### User:\nWhat is XSS?\n\n### Assistant:\n",
    "AES-256-GCM encrypts data at rest.",
]


def load_proto(vocab_file: Path):
    import os
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from transformers.utils import sentencepiece_model_pb2 as spb
    proto = spb.ModelProto()
    proto.ParseFromString(vocab_file.read_bytes())
    return proto


def extract_merges(pieces) -> list[tuple[str, str]]:
    """
    Recover BPE merges from SentencePiece piece scores.

    This mirrors the algorithm transformers uses internally: for every piece,
    collect EVERY split whose two halves are both in the vocab (not just the
    best one), order those candidates by the ids of their halves, then order all
    merges globally by the merged piece's score, descending.

    An earlier version kept only the single best split per piece and ordered by
    score alone. It produced a tokenizer that looked plausible and matched
    SentencePiece on 2 of 12 probes.
    """
    vocab = {p.piece: i for i, p in enumerate(pieces)}
    scores = {p.piece: p.score for p in pieces}

    merges: list[tuple[str, str, float]] = []
    for merged, piece_score in scores.items():
        local: list[tuple[str, str, float]] = []
        for index in range(1, len(merged)):
            piece_l, piece_r = merged[:index], merged[index:]
            if piece_l in vocab and piece_r in vocab:
                local.append((piece_l, piece_r, piece_score))
        local.sort(key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.extend(local)

    merges.sort(key=lambda val: val[2], reverse=True)
    return [(l, r) for l, r, _ in merges]


def build(vocab_file: Path):
    from tokenizers import Tokenizer, decoders, normalizers, processors
    from tokenizers.models import BPE

    proto = load_proto(vocab_file)
    pieces = list(proto.pieces)
    vocab = {p.piece: i for i, p in enumerate(pieces)}
    merges = extract_merges(pieces)

    tok = Tokenizer(BPE(
        vocab=vocab,
        merges=merges,
        unk_token="<unk>",
        fuse_unk=True,
        byte_fallback=bool(proto.trainer_spec.byte_fallback),
    ))
    # Prepend the word-boundary marker, THEN map spaces to it. Order matters.
    #
    # A Metaspace pre-tokenizer with prepend_scheme="always" ABSORBS a leading
    # space instead of placing a marker beside it: SentencePiece renders
    # " 63 rubber bands." as ['_ _', '63', ...] while Metaspace gives ['_63', ...].
    # ArithMark-3 endings all begin with a space and are tokenized separately
    # from the context, so that one difference changed 4,000 of 9,000 benchmark
    # strings and would have silently altered every score.
    steps = []
    # The tokenizer was trained with normalization_rule_name="nfkc"; without
    # this, accented and typographic characters tokenize differently.
    if "nfkc" in (proto.normalizer_spec.name or "").lower():
        steps.append(normalizers.NFKC())
    if proto.normalizer_spec.add_dummy_prefix:
        steps.append(normalizers.Prepend(prepend="▁"))
    steps.append(normalizers.Replace(pattern=" ", content="▁"))
    tok.normalizer = normalizers.Sequence(steps)
    # Decoder must mirror the normalizer AND undo byte fallback. Without
    # ByteFallback the <0xNN> pieces leak into the text verbatim, and without
    # Fuse the pieces are concatenated with the marker already stripped, which
    # silently deletes every space.
    tok.decoder = decoders.Sequence([
        decoders.Replace(pattern="▁", content=" "),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(content=" ", left=1, right=0),
    ])
    tok.post_processor = processors.TemplateProcessing(
        single="$A", pair="$A $B",
        special_tokens=[],
    )
    tok.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
    return tok, len(merges)


def verify(tok, vocab_file: Path) -> tuple[int, int, list[str]]:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(vocab_file))
    ok = 0
    failures = []
    for text in PROBES:
        truth = sp.encode(text, out_type=int)
        got = tok.encode(text).ids
        if got == truth:
            ok += 1
        else:
            failures.append(f"{text[:48]!r}\n      spm: {truth[:14]}\n      hf : {got[:14]}")
    return ok, len(PROBES), failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-file",
                    default=str(_REPO_ROOT / "tokenizer" / "tokenizer_output" / "tokenizer.model"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="write even if probes disagree with SentencePiece (do not use for benchmarks)")
    args = ap.parse_args()

    vocab_file = Path(args.vocab_file)
    if not vocab_file.exists():
        print(f"tokenizer not found: {vocab_file}", file=sys.stderr)
        return 1

    print("=" * 66)
    print("  SentencePiece BPE -> HuggingFace fast tokenizer")
    print("=" * 66)
    tok, n_merges = build(vocab_file)
    print(f"  vocab   : {tok.get_vocab_size():,}")
    print(f"  merges  : {n_merges:,}")

    ok, total, failures = verify(tok, vocab_file)
    print(f"  probes  : {ok}/{total} match SentencePiece exactly")
    for f in failures[:4]:
        print(f"    MISMATCH {f}")

    if ok != total and not args.allow_mismatch:
        print("\n  ABORT: tokenizer does not reproduce SentencePiece. Nothing written.",
              file=sys.stderr)
        print("  A wrong tokenizer does not crash a benchmark, it just scores the",
              file=sys.stderr)
        print("  model at chance -- which would be reported as a real result.",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 2048,
        "bos_token": "<bos>", "eos_token": "<eos>",
        "unk_token": "<unk>", "pad_token": "<pad>",
        "clean_up_tokenization_spaces": False,
        "added_tokens_decoder": {
            "0": {"content": "<pad>", "special": True, "lstrip": False, "rstrip": False,
                  "normalized": False, "single_word": False},
            "1": {"content": "<unk>", "special": True, "lstrip": False, "rstrip": False,
                  "normalized": False, "single_word": False},
            "2": {"content": "<bos>", "special": True, "lstrip": False, "rstrip": False,
                  "normalized": False, "single_word": False},
            "3": {"content": "<eos>", "special": True, "lstrip": False, "rstrip": False,
                  "normalized": False, "single_word": False},
        },
    }
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<bos>", "eos_token": "<eos>",
        "unk_token": "<unk>", "pad_token": "<pad>",
    }, indent=2), encoding="utf-8")

    print(f"\n  written to {out}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
