# CyberSLM — A Cybersecurity Small Language Model

A complete, from-scratch pipeline for a 33.5M-parameter decoder-only language model
specialised for cybersecurity — raw corpus → tokenizer → pretraining → instruction tuning →
evaluation, with every stage machine-checked.

**Weights:** [cyberslm-base](https://huggingface.co/sabari2005/cyberslm-base) (continuation) · [cyberslm-instruct](https://huggingface.co/sabari2005/cyberslm-instruct) (question answering)

This repository holds **code only** — the training corpus and model weights are not in git.

## Results

Retrained after fixing 19 defects. Both checkpoints scored by one process on **identical
windows at identical context**, 409,600 held-out tokens:

| metric | after fixes | before | change |
|---|---|---|---|
| validation loss | **2.3627** | 2.6255 | −0.263 |
| perplexity | **10.62** | 13.81 | **−23.1%** |
| top-1 accuracy | **57.21%** | 54.38% | +2.83 pp |
| top-5 accuracy | **72.64%** | 69.55% | +3.09 pp |
| 8-gram repetition | **23.7%** | 34.0% | −10.3 pp |

Both runs used the same 131,072 tokens/step. The previous run needed **4,000 steps (524M
tokens)** to reach a loss this one passed at **750 steps (98M tokens)** — the old dataloader
sampled windows *with replacement* and its per-epoch reshuffle never reached the DataLoader
workers, so every epoch replayed an identical ~63% of the corpus.

Trained on one A100-40GB: 786M tokens in 71 min at 184,084 tok/s, ~$3.65 total including
instruction tuning.

**Honest caveat.** At 33.5M parameters the instruction-tuned model learned the *shape* of a
good answer and is frequently wrong about the *content*. Only 2 of 8 evaluation prompts
stopped on EOS by themselves. See [`runs/reports/FINAL_REPORT.md`](runs/reports/FINAL_REPORT.md)
for the measurements and the failure cases, not just the good ones.

---


## Repository layout

```
SLm Dataset/
├── README.md                     ← this file
├── requirements.txt              ← consolidated Python dependencies
├── .gitignore
├── docs/
│   ├── CODE_REVIEW_AND_FIXES.md  ← bug review + fixes
│   └── NEW_ARCHITECTURE.md       ← proposed <100M architecture (CyberSLM‑2)
│
├── Dataset_corpus/               ← raw curated JSONL corpus (by domain)
│   ├── Cybersecurity/            ← 16 security sub‑domains
│   ├── Programming/              ← 8 languages / tooling
│   ├── Computer_Fundamentals/
│   ├── General_English/
│   └── Merge.py                  ← merges the corpus into one JSONL
│
├── tokenizer/                    ← SentencePiece BPE tokenizer (vocab 32k)
│   ├── train_tokenizer.py        ← trains tokenizer.model
│   ├── preprocessing.py          ← corpus text extraction
│   └── ...                       ← stats / counters / tests
│
├── Preprocessing_Pipeline/       ← corpus → packed uint16 token bins
│   ├── dataset_tokenizer.py      ← JSONL → tokens.bin (+ EOS per doc)
│   ├── dataset_builder.py        ← tokens.bin → train.bin / val.bin (95/5)
│   ├── dataloader.py             ← mmap DataLoader for pretraining
│   └── ...                       ← stats / verify
│
├── cyberslm/                     ← STAGE 1: pretraining (the base model)
│   ├── model/                    ← config, RoPE, RMSNorm, SwiGLU, attention, block
│   ├── training/                 ← trainer, scheduler, checkpoint, metrics
│   ├── train.py                  ← pretraining entry point (argparse; --dry-run)
│   ├── inference.py              ← base‑model text generation (KV‑cached)
│   └── scripts/
│       ├── verify.py             ← 29 correctness checks, CPU‑only, no training
│       └── evaluate.py           ← held‑out metrics + baseline A/B comparison
│
├── infra/                        ← Modal (serverless GPU) training
│   ├── modal_app.py              ← smoke / pretrain / sft / ls entry points
│   ├── upload_data.py            ← push corpus + tokenizer to the data volume
│   └── download_model.py         ← pull trained checkpoints back
│
├── runs/                         ← all training output (gitignored)
│
└── cyberslm_sft/                 ← STAGE 2: supervised fine‑tuning (instruct)
    ├── model/cyberslm.py         ← imports the Stage‑1 architecture verbatim
    ├── data/                     ← chat template, loss masking, collator, dataset
    ├── configs/sft_config.py     ← all SFT hyperparameters
    ├── utils/                    ← optimizer, validation, checkpoint, inference
    ├── train.py                  ← SFT entry point
    ├── evaluate.py               ← standalone checkpoint evaluation
    └── inference.py              ← chat with the fine‑tuned model
```

`cyberslm` and `cyberslm_sft` are the two Python packages. Run scripts from the repo root; the
`cyberslm_sft` entry points bootstrap their own `sys.path`, so they work from any directory.

---

## Quick start

```bash
# 0. run a trained model (download weights from Hugging Face first)
python infer_chat.py --checkpoint models/instruct.pt --prompt "What is SQL injection?"
python infer_base.py --checkpoint models/base.pt   --prompt "SQL injection is"

# 1. verify the model + data path is correct (CPU, seconds, no training)
python cyberslm/scripts/verify.py

# 2. check a training config without starting a run
python cyberslm/train.py --dry-run --seq-len 2048 --steps 6000

# 3. train on a GPU via Modal
python infra/upload_data.py                      # one-time, ~490 MB
modal run infra/modal_app.py::smoke              # ~1 min of GPU, proves the path
modal run --detach infra/modal_app.py::pretrain  # the real run
python infra/download_model.py                   # pull best.pt back

# 4. measure it, against the previous checkpoint on identical batches
python cyberslm/scripts/evaluate.py     --checkpoint runs/base/best.pt     --baseline cyberslm/checkpoints/best.pt     --context 2048
```

> **Do not run `python cyberslm/train.py` with no arguments on a laptop.** It starts a real
> multi-hour run. (It used to do this even for `--help`; that is fixed, but the warning stands.)

---

## Architecture (base model)

Decoder‑only pre‑norm transformer (LLaMA‑class):

| Property | Value |
|---|---|
| Parameters | ~33.5M |
| Hidden dim | 384 |
| Layers | 12 |
| Attention heads | 6 (head dim 64) |
| FFN | SwiGLU, inner 1024 |
| Vocab | 32,000 (SentencePiece BPE, byte‑fallback) |
| Context | 4,096 (RoPE, base 10,000) |
| Norm | RMSNorm |
| Weight tying | embedding ↔ LM head |

Attention uses `torch.nn.functional.scaled_dot_product_attention` (FlashAttention when available) with
an optional key‑padding mask; causality is applied without materialising a `T×T` buffer.

---

## Pipeline

```
Dataset_corpus/*.jsonl
   │  Merge.py + tokenizer/preprocessing.py
   ▼
one clean corpus (text per line)
   │  tokenizer/train_tokenizer.py            → tokenizer.model (vocab 32k)
   ▼
Preprocessing_Pipeline/dataset_tokenizer.py   → tokens.bin (+ <eos> per document)
   │  dataset_builder.py (95/5 doc‑level split)
   ▼
train.bin / val.bin (flat uint16 streams)
   │  cyberslm/train.py                        → base checkpoint (best.pt)
   ▼
cyberslm_sft/train.py (SFT on SFT.jsonl)       → instruct checkpoint (checkpoints/best/)
```

---

## Quickstart

```bash
# 0. install
pip install -r requirements.txt

# 1. merge + clean the corpus
python Dataset_corpus/Merge.py

# 2. train the tokenizer (vocab 32k, byte-fallback, split-digits)
python tokenizer/train_tokenizer.py --input tokenizer/Final.jsonl \
    --output-dir tokenizer/tokenizer_output

# 3. tokenize to a binary cache (EOS between documents is ON by default)
python Preprocessing_Pipeline/dataset_tokenizer.py \
    --input tokenizer/Final.jsonl \
    --model tokenizer/tokenizer_output/tokenizer.model \
    --output-dir tokenizer/cache

# 4. build train.bin / val.bin (deterministic 95/5 split)
python Preprocessing_Pipeline/dataset_builder.py \
    --cache-dir tokenizer/cache --output-dir tokenizer/data

# 5. pretrain the base model  (needs a GPU)
python cyberslm/train.py

# 6. supervised fine-tune  (needs a GPU + the pretrained checkpoint copied in)
cp cyberslm/checkpoints/best.pt cyberslm_sft/checkpoints/pretrained/model.pt
cp tokenizer/tokenizer_output/tokenizer.model cyberslm_sft/tokenizer/tokenizer.model
python cyberslm_sft/train.py

# 7. chat with the fine-tuned model
python cyberslm_sft/inference.py --interactive
```

---

## Special tokens (important)

The tokenizer defines real control ids that are used **by id**, never as literal strings:

| id | piece | role |
|---|---|---|
| 0 | `<pad>` | padding |
| 1 | `<unk>` | unknown (rare — byte‑fallback covers most) |
| 2 | `<bos>` | beginning of sequence |
| 3 | `<eos>` | end of sequence / document / assistant turn |

Every document (pretraining) and every assistant turn (SFT) ends with `<eos>` (id 3), and generation
stops on id 3. Do **not** inject literal `"<s>"`/`"</s>"` text — see the review doc for why this
previously broke stopping.

---

## Reproducing / regenerating artifacts

Large binary artifacts (`*.bin`, `*.idx`, `tokenizer.model`, checkpoints, `Final.jsonl`, `SFT.jsonl`)
are git‑ignored and regenerated by the pipeline above. The small curated corpus under
`Dataset_corpus/` is the source of truth.

---

## License / use

Educational and defensive‑security use. Security content is dual‑use; keep to defensive framing and
follow the ethics notes in `docs/NEW_ARCHITECTURE.md`.
