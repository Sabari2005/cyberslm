# Dataset Processing Pipeline

**Stage:** Dataset Processing  
**Part of:** Cybersecurity-focused decoder-only Small Language Model (SLM)

This pipeline transforms a raw JSONL dataset into binary token files ready for transformer pretraining. It sits immediately after tokenizer training and immediately before model training.

```
Tokenizer training (completed)
        ↓
Dataset Processing  ◄── this pipeline
        ↓
Transformer Architecture
        ↓
Pretraining
```

---

## Prerequisites

```
tokenizer.model      ← trained SentencePiece BPE model
tokenizer.vocab      ← vocabulary file
dataset.jsonl        ← input dataset (one {"text": "..."} per line)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

```
dataset.jsonl
     │
     ▼
1. tokenizer_analyzer.py    Inspect tokenizer, verify round-trips
     │
     ▼
2. dataset_tokenizer.py     Stream JSONL → encode → tokens.bin + tokens.idx
     │
     ▼
3. dataset_stats.py         Compute token/document statistics
     │
     ▼
4. dataset_builder.py       Split 95/5 → train.bin + val.bin
     │
     ▼
5. dataloader.py            Memory-mapped DataLoader for training
     │
     ▼
6. verify.py                End-to-end pipeline verification
```

---

## Step-by-Step Usage

### Step 1 — Analyze the tokenizer

```bash
python tokenizer_analyzer.py \
    --model tokenizer.model \
    --vocab tokenizer.vocab \
    --top-k 50
```

Outputs a report to stdout covering:
- Vocabulary size and piece type breakdown
- Special token IDs (BOS, EOS, UNK, PAD)
- Average token length
- Top-K vocabulary pieces by BPE score
- Character coverage estimate
- Encode→decode round-trip verification

---

### Step 2 — Tokenize the dataset

```bash
python dataset_tokenizer.py \
    --input  dataset.jsonl \
    --model  tokenizer.model \
    --output-dir ./cache \
    --add-bos \
    --add-eos
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Path to `dataset.jsonl` |
| `--model` | required | Path to `tokenizer.model` |
| `--output-dir` | `./cache` | Output directory |
| `--workers` | `cpu_count - 1` | Parallel encoding workers |
| `--batch-size` | `1024` | Documents per encoding batch |
| `--add-bos` | off | Prepend BOS token per document |
| `--add-eos` | off | Append EOS token per document |

**Output files:**

| File | Description |
|------|-------------|
| `cache/tokens.bin` | Flat uint16 token IDs, little-endian |
| `cache/tokens.idx` | uint64 byte offsets per document (sentinel final entry) |
| `cache/tokenize.log` | Processing summary |

**Error handling:** Invalid JSON, missing `text` fields, empty documents, and Unicode errors are all silently skipped and counted. The log shows a skip breakdown by reason.

---

### Step 3 — Compute dataset statistics

```bash
python dataset_stats.py \
    --cache-dir ./cache \
    --context-len 4096 \
    --histogram-bins 20 \
    --output stats.json
```

Outputs to stdout:
- Document counts (total, valid, empty)
- Token statistics (mean, median, std, min, max, P25, P75, P95, P99)
- Documents exceeding context length
- ASCII histogram of document lengths
- Token-range bucket distribution

Optionally writes all statistics to `--output` as JSON.

---

### Step 4 — Build train/val binary files

```bash
python dataset_builder.py \
    --cache-dir  ./cache \
    --output-dir ./data \
    --val-ratio  0.05 \
    --seed       42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--val-ratio` | `0.05` | Fraction of documents for validation |
| `--seed` | `42` | RNG seed for deterministic split |
| `--chunk-size` | `67108864` | Write buffer size in bytes (64 MB) |

**Output files:**

| File | Description |
|------|-------------|
| `data/train.bin` | Flat uint16 training token stream |
| `data/val.bin` | Flat uint16 validation token stream |
| `data/split.json` | Document indices, token counts, seed (for reproducibility) |

The split is performed at document boundaries, so no document straddles train and val. Within each file, tokens are stored sequentially with no gaps. The DataLoader applies a sliding window at training time.

---

### Step 5 — Use the DataLoader

```python
from pathlib import Path
from dataloader import make_train_dataloader, make_val_dataloader

# Training
train_loader, train_dataset = make_train_dataloader(
    bin_path=Path("data/train.bin"),
    context_len=4096,
    batch_size=8,
    num_samples=500_000,   # samples per epoch
    seed=42,
    num_workers=4,
)

# Call before each epoch to reshuffle start positions
train_dataset.set_epoch(epoch)

for x, y in train_loader:
    # x: (batch_size, context_len) int64
    # y: (batch_size, context_len) int64  where y[b,i] == x[b,i+1]
    loss = model(x, targets=y)

# Validation (sequential, non-overlapping windows)
val_loader = make_val_dataloader(
    bin_path=Path("data/val.bin"),
    context_len=4096,
    batch_size=8,
    num_workers=2,
)
```

**Multi-GPU (DDP):**

```python
train_loader, _ = make_train_dataloader(
    ...,
    distributed=True,
    rank=local_rank,
    world_size=world_size,
)
```

**DataLoader internals:**

| Feature | Implementation |
|---------|---------------|
| RAM usage | O(1) — memory-mapped file, tensors only |
| Start positions | Pre-sampled numpy array, refreshed per epoch |
| Shuffle | Per-epoch via seeded RNG; DistributedSampler in DDP mode |
| Pinned memory | Enabled when CUDA is available |
| Multiple workers | Persistent workers with prefetch_factor=2 |
| Drop last | True for training (avoids partial batches) |

---

### Step 6 — Verify the pipeline

```bash
python verify.py \
    --model     tokenizer.model \
    --cache-dir ./cache \
    --data-dir  ./data
```

Runs 7 independent checks:

| Check | Description |
|-------|-------------|
| `roundtrip` | Encode → decode returns original text |
| `cache` | tokens.bin and tokens.idx are internally consistent |
| `split_files` | train.bin, val.bin, split.json are consistent |
| `determinism` | Re-generating the split with the stored seed is identical |
| `token_counts` | train+val token count ≤ cache token count |
| `compat` | Vocabulary fits in uint16; BOS/EOS are valid |
| `dataloader` | `y[i] == x[i+1]` for 20 random samples |

Exit code 0 = all passed. Exit code 1 = at least one failure.

---

## Binary File Format

```
tokens.bin
┌──────────────────────────────────────────────────────────┐
│  token_0  │  token_1  │  token_2  │  …  │  token_N-1   │
│  uint16LE │  uint16LE │  uint16LE │     │   uint16LE    │
└──────────────────────────────────────────────────────────┘
                      2 bytes per token

tokens.idx  (cache only; not needed after train.bin/val.bin are built)
┌──────────────────────────────────────────────────────────┐
│  offset_0 │  offset_1 │  …  │  offset_N-1 │  offset_N  │
│  uint64LE │  uint64LE │     │   uint64LE  │  uint64LE  │
└──────────────────────────────────────────────────────────┘
  N+1 entries for N documents; final entry = total bytes in tokens.bin
  Document i occupies bytes [offset_i, offset_{i+1}) in tokens.bin
  Token count = (offset_{i+1} - offset_i) / 2
```

---

## File Structure

```
slm_data_pipeline/
├── tokenizer_analyzer.py   Tokenizer inspection and round-trip verification
├── dataset_tokenizer.py    JSONL → binary token cache
├── dataset_stats.py        Token and document statistics
├── dataset_builder.py      Train/val split and binary file builder
├── dataloader.py           Memory-mapped PyTorch DataLoader
├── verify.py               End-to-end pipeline verification
├── requirements.txt        Python dependencies
└── README.md               This file

cache/                      (created by dataset_tokenizer.py)
├── tokens.bin
├── tokens.idx
└── tokenize.log

data/                       (created by dataset_builder.py)
├── train.bin
├── val.bin
└── split.json
```

---

## Performance Notes

- **Tokenization throughput** scales linearly with `--workers`. Use `cpu_count - 1` for best results. Each worker holds its own SentencePiece instance.
- **Binary I/O** uses large write buffers (default 64 MB) to minimise syscall overhead during `dataset_builder.py`.
- **DataLoader** uses `mmap.ACCESS_READ` (OS-level demand paging). The OS will cache hot pages in the page cache automatically.
- **Multiple DataLoader workers** each open their own mmap handle; no locking or IPC required.
- For very large datasets (>500 GB), set `--workers` in `dataset_tokenizer.py` to match available CPU cores and ensure the output directory is on a fast local SSD.

---

## Reproducibility

The entire pipeline is deterministic given the same:
- `tokenizer.model`
- Input `dataset.jsonl`
- `--seed` value
- `--val-ratio` value
- `--add-bos` / `--add-eos` flags

The `split.json` file records all parameters needed to reproduce the exact split.
