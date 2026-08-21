"""
train_tokenizer_v2.py
=====================
Trains the SentencePiece BPE tokenizer for CyberSLM-2.

Differences from the v1 tokenizer, and why each matters:

  vocab 32768 (was 32000)
      A power of two keeps the final GEMM and the softmax on tensor-core-friendly
      shapes. The extra 768 slots absorb the control tokens and reserves below.

  control tokens as user_defined_symbols
      <|user|>, <|tool_call|>, <|think|> and friends become single atomic ids.
      Without this they would be shredded into "<", "|", "user", "|", ">" and the
      model would have to learn a five-token idiom instead of one embedding --
      and, worse, could emit a near-miss that no parser recognizes.

  byte_fallback
      Guarantees any byte sequence is encodable. Essential for a security corpus
      full of hex dumps, base64 blobs, shellcode and malformed input.

  split_digits
      Forces "4096" to tokenize as 4/0/9/6 rather than one merged piece. Digit
      merges are the standard reason small models fail at arithmetic: the model
      cannot do column-wise math on a token that hides the columns.

  allow_whitespace_only_pieces + remove_extra_whitespaces=False
      Indentation is semantic in Python and YAML. Collapsing runs of spaces
      destroys the structure the model is meant to learn.

Usage
-----
    python -m cyberslm2.scripts.train_tokenizer_v2 \\
        --input tokenizer/Final.jsonl \\
        --output-dir tokenizer/v2

Training a 32k vocab over a ~780MB corpus needs roughly 20-30 GB of RAM at
default settings; --input-sentence-size caps the sample used to control that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cyberslm2.data.special_tokens import (
    BOS,
    CONTROL_TOKENS,
    EOS,
    N_RESERVED,
    PAD,
    UNK,
)


def extract_corpus(jsonl_path: Path, out_path: Path, text_key: str = "text") -> int:
    """Flatten a JSONL corpus to one raw text line per document."""
    n = 0
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_key) or obj.get("content") or ""
            if isinstance(text, str):
                text = text.replace("\n", " ").strip()
                if text:
                    fout.write(text + "\n")
                    n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Train the CyberSLM-2 tokenizer")
    p.add_argument("--input", required=True, help="corpus .jsonl or raw .txt")
    p.add_argument("--output-dir", default="tokenizer/v2")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--text-key", default="text")
    p.add_argument("--character-coverage", type=float, default=0.9999)
    p.add_argument("--input-sentence-size", type=int, default=5_000_000)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="print the token layout and the exact trainer args, then exit")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    prefix = out_dir / "tokenizer"

    # ids 4..4+len-1 are the control tokens; then padding filler to N_RESERVED
    reserved = [f"<|reserved_{i}|>" for i in range(N_RESERVED - len(CONTROL_TOKENS))]
    user_defined = CONTROL_TOKENS + reserved

    print("=" * 68)
    print("CyberSLM-2 tokenizer v2")
    print("=" * 68)
    print(f"  vocab_size        {args.vocab_size}")
    print(f"  control tokens    {len(CONTROL_TOKENS)}")
    print(f"  reserved slots    {len(reserved)}")
    print("\n  id layout:")
    print(f"    0 {PAD}   1 {UNK}   2 {BOS}   3 {EOS}")
    for i, tok in enumerate(CONTROL_TOKENS):
        print(f"    {4 + i:<3} {tok}")
    if reserved:
        print(f"    {4 + len(CONTROL_TOKENS)}..{3 + len(user_defined)} "
              f"<|reserved_*|>")
    print("=" * 68)

    if args.dry_run:
        print("\n--dry-run: no corpus was read and no tokenizer was trained.")
        return 0

    import sentencepiece as spm

    src = Path(args.input)
    if not src.exists():
        print(f"ERROR: input not found: {src}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    if src.suffix == ".jsonl":
        corpus = out_dir / "corpus.txt"
        print(f"Extracting text from {src} -> {corpus} ...")
        n = extract_corpus(src, corpus, args.text_key)
        print(f"  wrote {n:,} documents")
    else:
        corpus = src

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        num_threads=args.num_threads,

        # --- control ids, fixed by position ---
        pad_id=0, pad_piece=PAD,
        unk_id=1, unk_piece=UNK,
        bos_id=2, bos_piece=BOS,
        eos_id=3, eos_piece=EOS,
        user_defined_symbols=user_defined,

        # --- the choices that matter for code and security text ---
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        max_sentencepiece_length=16,
    )

    print(f"\nWrote {prefix}.model and {prefix}.vocab")

    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
    from cyberslm2.data.special_tokens import SpecialTokens
    SpecialTokens.validate(sp)
    print(f"Validated against the token protocol. vocab={sp.get_piece_size()}")

    sample = 'Run nmap -sV 10.0.0.1 then parse CVE-2021-44228 (log4j) output.'
    ids = sp.encode(sample, out_type=int)
    print(f"\nSample round-trip ({len(ids)} tokens):\n  {sample}\n  -> "
          f"{sp.decode(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
