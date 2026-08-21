# Cybersecurity SLM Tokenizer

Production tokenizer training pipeline for a decoder-only transformer SLM
specialized in cybersecurity, with general English, programming, and
computer science coverage. Trains a 32,000-piece SentencePiece BPE model
directly from a JSONL text corpus — no pretrained tokenizer or model is
used anywhere in this pipeline.

This tokenizer is model-size-agnostic: the same `tokenizer.model` is
reused unchanged across every future model scale (10M → 50M → 100M →
500M parameters). Vocabulary and parameter count are independent
concerns — only retrain the tokenizer if the *data domain* changes
significantly, not when the *model* changes size.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Input Format](#input-format)
- [How to Run](#how-to-run)
- [Configuration Reference](#configuration-reference)
- [Design Decisions Beyond the Base Spec](#design-decisions-beyond-the-base-spec)
- [Output Files](#output-files)
- [Expected Behavior](#expected-behavior)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Extending the Tokenizer Later](#extending-the-tokenizer-later)

## Overview

The pipeline has two stages, run sequentially by `train_tokenizer.py`:

1. **Extraction** (`preprocessing.py`) — streams the input JSONL file,
   validates and cleans each record, and writes one training document per
   line to an intermediate corpus file. JSON parsing and validation are
   distributed across worker processes; the input file is never fully
   loaded into memory.
2. **Training** (`train_tokenizer.py`) — invokes the SentencePiece BPE
   trainer on the cleaned corpus, then loads the resulting model and
   runs an encode/decode sanity check on representative cybersecurity,
   code, and general-text strings before reporting success.

The intermediate corpus file is deleted automatically after training
unless `--keep-corpus` is passed.

## Requirements

- Python 3.9+
- `sentencepiece>=0.2.0` (the only dependency; see `requirements.txt`)
- Multi-core CPU recommended for large corpora (the pipeline scales
  preprocessing and training across all available cores by default)

## Installation

```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── README.md
├── requirements.txt
├── preprocessing.py      # streaming JSONL extraction/validation
└── train_tokenizer.py    # CLI entry point: orchestration + training + validation
```

Running the pipeline additionally produces, inside `--output-dir`:

```
tokenizer_output/
├── tokenizer.model        # binary SentencePiece model (used at train/inference time)
└── tokenizer.vocab        # human-readable piece list with scores, for inspection
```

## Input Format

A single JSONL file, one JSON object per line, UTF-8 encoded:

```json
{"text": "CVE-2024-3400 affects PAN-OS GlobalProtect gateways."}
{"text": "def scan_ports(host: str, ports: list[int]) -> dict:\n    return {}"}
```

Only the `text` field is read. Any other fields present on a line are
ignored, not an error.

## How to Run

```bash
python train_tokenizer.py \
    --input /path/to/corpus.jsonl \
    --output-dir ./tokenizer_output
```

PowerShell (Windows): use a single line, or use PowerShell's line
continuation character (backtick: `` ` ``), not backslash.

```powershell
# single line (recommended)
python train_tokenizer.py --input .\ComputerFundamentals_dataset.jsonl --output-dir .\tokenizer_output

# multiline in PowerShell
python train_tokenizer.py `
  --input .\ComputerFundamentals_dataset.jsonl `
  --output-dir .\tokenizer_output
```

This uses the exact configuration specified for this project (32,000
vocab, BPE, NFKC, 0.9995 character coverage, byte fallback, case
preserved). All CLI flags have production-ready defaults — for a
standard run, `--input` is the only required argument.

Useful overrides:

```bash
# Cap threads/processes explicitly (defaults to all available cores)
python train_tokenizer.py --input corpus.jsonl --output-dir ./out --num-threads 8 --workers 8

# Keep the intermediate extracted-text corpus for inspection or reuse
python train_tokenizer.py --input corpus.jsonl --output-dir ./out --keep-corpus

# Use every line in the corpus instead of the 20M-line sample cap
python train_tokenizer.py --input corpus.jsonl --output-dir ./out --input-sentence-size 0
```

Run `python train_tokenizer.py --help` for the full flag list.

## Configuration Reference

Every trainer option this pipeline sets, and what it does:

| Option | Value | Purpose |
|---|---|---|
| `model_type` | `bpe` | Byte pair encoding, as specified. |
| `vocab_size` | `32000` | Target piece count. |
| `hard_vocab_limit` | `True` | Enforces exactly `vocab_size` pieces rather than stopping early. |
| `character_coverage` | `0.9995` | Fraction of corpus characters guaranteed their own piece; the long tail of the rarest 0.05% falls back to byte encoding instead of consuming vocab budget. |
| `normalization_rule_name` | `nfkc` | Unicode NFKC normalization. Case is untouched — NFKC does not fold case (unlike `nfkc_cf`). |
| `byte_fallback` | `True` | Any byte sequence SentencePiece hasn't seen enough to learn a piece for (rare Unicode, raw binary pasted as text, unusual encodings) is still representable via the 256 `<0xXX>` byte tokens instead of collapsing to `<unk>` and losing information. |
| `remove_extra_whitespaces` | `False` | Preserves exact whitespace runs — critical for indentation-sensitive code (Python, YAML) and aligned log output. |
| `allow_whitespace_only_pieces` | `True` | Lets the trainer learn dedicated pieces for pure-whitespace runs (e.g. a 4-space indent as one token) instead of forcing whitespace into single-character pieces. |
| `add_dummy_prefix` | `True` | Prepends a virtual leading space to every training line so a word tokenizes identically whether it starts a line or follows a space — standard SentencePiece practice. |
| `split_digits` | `False` | Digits are not forced into single-character pieces, so common numeric substrings (port numbers, years, octets) can merge into efficient multi-digit tokens. |
| `split_by_number` | `False` | See [below](#design-decisions-beyond-the-base-spec) — this is the one setting changed from SentencePiece's own default, with measured justification. |
| `max_sentencepiece_length` | `16` | Longest allowed single piece, in Unicode chars (SentencePiece's own default). |
| `max_sentence_length` | `16384` bytes | Longest allowed training line, raised from SentencePiece's 4192-byte default to avoid truncating long single-line technical content (long URLs, minified JSON, base64 blobs). |
| `input_sentence_size` | `20000000` | Caps how many lines are loaded for training (0 = unlimited). See below. |
| `shuffle_input_sentence` | `True` | When the cap above applies, sample uniformly across the whole file rather than just its first N lines. |
| `train_extremely_large_corpus` | `True` | Switches internal counters to a mode safe for corpora large enough to overflow default counters — matches the "hundreds of millions of tokens" scale target. |
| `num_threads` | all available cores | Parallelizes the trainer's internal computation. |
| `pad_id` / `pad_piece` | `0` / `<pad>` | |
| `unk_id` / `unk_piece` | `1` / `<unk>` | |
| `bos_id` / `bos_piece` | `2` / `<bos>` | |
| `eos_id` / `eos_piece` | `3` / `<eos>` | |

## Design Decisions Beyond the Base Spec

Everything explicitly specified (BPE, 32000, NFKC, 0.9995 coverage,
case-preserving, byte fallback, the four special tokens) is implemented
exactly as given. A few additional trainer options were set because
they have a clear, verifiable technical justification for this domain
and don't conflict with anything specified:

**`split_by_number=False` (SentencePiece's own default is `True`).**
By default SentencePiece forbids any merge that crosses a digit/letter
boundary. This is usually reasonable for prose, but it's actively harmful
for hex content — and cybersecurity text is full of it (MD5/SHA1/SHA256
hashes, hex-encoded shellcode, memory addresses). I trained matched test
tokenizers with this flag both ways on a hash-containing sample: a
64-character SHA-256 string encoded as **29 pieces** with the default
`True`, versus **8 pieces** with `False` (the difference is entirely
merges like a 14-character contiguous hex run becoming one piece instead
of fragmenting at every letter/digit transition). IP addresses, CVE IDs,
and port numbers were tokenized identically either way, so this change
only helps and doesn't trade anything off. This is the one place this
pipeline deviates from SentencePiece's own out-of-the-box default —
everything else in the table above is either the literal spec or
SentencePiece's existing default made explicit.

**`normalization_rule_name=nfkc` instead of SentencePiece's own default
(`nmt_nfkc`).** `nmt_nfkc` layers additional NMT-oriented whitespace and
punctuation normalization on top of plain NFKC. Plain `nfkc` was chosen
because the spec calls for exactly NFKC, and the extra NMT-specific
normalization is undesirable here regardless — it's tuned for
translation corpora, not for preserving exact technical text.

**`remove_extra_whitespaces=False` and `allow_whitespace_only_pieces=True`
(SentencePiece's own defaults are `True` / `False`).** Left at their
defaults, SentencePiece collapses runs of spaces/tabs to a single space
during training. That silently destroys Python and YAML indentation and
column-aligned log/table output — a correctness problem for this domain,
not a style preference. Verified empirically: with both changes applied,
a 4-space indent and an 8-space indent each tokenize as a single piece;
with the defaults, multi-space runs collapse to one space before the
tokenizer ever sees them.

**`input_sentence_size=20000000` (SentencePiece's own default is `0`,
unlimited).** SentencePiece's trainer loads its full sampled sentence set
into memory at once — this is an architectural property of the trainer,
not a choice made here. For a corpus of "hundreds of millions" of lines,
unbounded loading risks unpredictable, very large memory use. 20M
shuffled lines is far more than BPE needs to converge on stable merge
statistics at a 32k vocabulary, and if your corpus has fewer than 20M
valid lines, every line is used and this cap has no effect. Pass
`--input-sentence-size 0` to force using every line regardless of corpus
size if you have the memory budget for it.

**`max_sentence_length=16384` (SentencePiece's own default is `4192`
bytes).** Raised so that long single-line technical content — long URLs,
minified JSON blobs, base64 payloads — isn't silently dropped by the
trainer for exceeding the line-length cap.

## Output Files

- **`tokenizer.model`** — binary SentencePiece model. Load with
  `sentencepiece.SentencePieceProcessor(model_file="tokenizer.model")`
  for both training-time and inference-time tokenization.
- **`tokenizer.vocab`** — tab-separated `piece\tscore` listing, one per
  line, in ID order. Useful for manual inspection; not required at
  runtime.

## Expected Behavior

A full run logs, in order: a preprocessing summary (lines read, valid
documents, and a breakdown of any skipped lines by reason), SentencePiece's
own verbose training log (it logs directly to stderr; this is the
library's normal behavior, not an error), then a post-training validation
report that loads the model and encode/decode-round-trips it against
sample CVE IDs, IPs, URLs, SQL, PowerShell, Python, JSON, and YAML
strings. A "Done" line with the output path confirms success. Exit code
is `0` on success, `1` on a handled error, `130` if interrupted.

## Reproducibility

The extraction stage is fully deterministic: the same input file always
produces a byte-identical intermediate corpus, independent of
`--workers` (verified — 1, 2, and 4 workers produce identical output).
Line order is preserved rather than written in worker-completion order,
specifically so this holds.

The SentencePiece training stage itself is *not* guaranteed bit-for-bit
reproducible between separate runs, even with an identical corpus and
identical configuration — this was verified directly, including with
`shuffle_input_sentence=False` and `num_threads=1`. The trainer's C++
implementation doesn't expose a settable seed for the internal
tie-breaking it does among equally-frequent merge candidates. In
practice this affects only a small number of low-frequency merges at the
end of training; vocabulary size, special tokens, and overall tokenization
behavior are unaffected. If you need a fixed reference tokenizer, treat
the specific `tokenizer.model` file you trained as the canonical
artifact rather than re-running training and expecting an identical file.

## Troubleshooting

**`RuntimeError` during training mentioning vocab size** — this cuts both
ways. Either the corpus has too little unique text to reach the
requested `--vocab-size` (add more data or lower `--vocab-size`), or
`--vocab-size` is set too low to even fit the fixed overhead: 4 special
tokens + 256 byte-fallback tokens + the corpus's alphabet size, roughly
260+ pieces as a practical floor whenever `byte_fallback` is enabled (raise
`--vocab-size`). The default of 32000 is far above this floor and won't
hit it in practice; it only shows up if you lower `--vocab-size` a lot for
quick experimentation.

**Preprocessing reports a large number of `invalid_unicode` or
`invalid_json` lines** — check that the source file is genuinely UTF-8
and that it's valid JSONL (one complete JSON object per line, not a
single pretty-printed JSON array). A small number of skipped lines from
a scraped/aggregated dataset is normal; a large fraction indicates a
mismatched file format or encoding.

**Training is slow or uses more memory than expected** — lower
`--input-sentence-size`, or explicitly cap `--num-threads`/`--workers`
if running alongside other jobs on a shared machine.

**`pip install sentencepiece` fails** — the PyPI package ships prebuilt
wheels for common platforms; a build-from-source failure usually means
missing a C++ toolchain. Installing on a standard Linux x86_64 or macOS
environment with an up-to-date `pip` avoids this in virtually all cases.

**Output directory already contains a `tokenizer.model`** — it's
overwritten; SentencePiece does not merge with or version existing
models.

## Extending the Tokenizer Later

Instruction tuning and tool-calling will eventually need additional
special tokens (turn/role markers, tool-call delimiters) beyond the four
trained here. This pipeline intentionally doesn't guess at that format
now, since it isn't specified. When it's needed, the two standard paths
are: (1) retrain from the same corpus with the new tokens added via
SentencePiece's `control_symbols`/`user_defined_symbols` trainer options,
keeping everything else in this config identical for continuity, or (2)
append new token IDs after `vocab_size` and extend the model's embedding
matrix at fine-tuning time, which is standard practice and doesn't
require retraining the base tokenizer. Either approach is compatible
with everything trained by this pipeline as-is.
