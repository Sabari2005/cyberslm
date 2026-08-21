# CyberSLM — Supervised Instruction Fine-Tuning Pipeline

Production-quality SFT pipeline for training **CyberSLM-Instruct** from the
pretrained **CyberSLM** base model.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Project Structure](#project-structure)
3. [Dataset Format](#dataset-format)
4. [Prompt Template](#prompt-template)
5. [Loss Masking](#loss-masking)
6. [Training Workflow](#training-workflow)
7. [Resume Training](#resume-training)
8. [Hyperparameters](#hyperparameters)
9. [Expected Hardware](#expected-hardware)
10. [Troubleshooting](#troubleshooting)

---

## Pipeline Overview

```
Pretrained CyberSLM Base Model (33.53 M parameters)
              │
              ▼
  ┌───────────────────────┐
  │   Dataset Loader      │  Reads JSONL; supports alpaca + conversation formats
  │   Dataset Validator   │  Drops malformed samples; strict or lenient mode
  │   Prompt Formatter    │  Tokenises + applies loss masking
  │   Conversation Template│  Serialises multi-turn conversations
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │   SFT Collator        │  Pads batches; preserves -100 labels
  │   Loss Masking        │  Cross-entropy only on assistant response tokens
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │   SFTTrainer          │  AdamW + cosine LR + gradient accumulation
  │   Validation Loop     │  Loss / perplexity / tok/s every N steps
  │   Checkpoint Manager  │  latest/ best/ step_N/ with full resume
  └───────────────────────┘
              │
              ▼
  CyberSLM-Instruct
  (instruction-following fine-tuned model)
              │
              ▼
  Inference Sanity Checks (6 prompts: SQL injection, AES, Python, CIA triad, Linux, Summarise)
```

**Objective**: Teach the pretrained model *behaviour* — instruction following,
Q&A, reasoning, code generation, summarisation — without injecting new
factual knowledge.  All knowledge comes from pretraining; SFT only refines
the response format and instruction adherence.

---

## Project Structure

```
cyberslm_sft/
├── train.py                    # Main training entry point
├── evaluate.py                 # Standalone evaluation script
├── configs/
│   ├── __init__.py
│   └── sft_config.py           # All hyperparameters (SFTConfig dataclass)
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py       # JSONL loader; normalises both sample formats
│   ├── dataset_validator.py    # Per-sample validation with error reporting
│   ├── prompt_formatter.py     # Tokenises + applies loss mask boundaries
│   ├── conversation_template.py# String serialiser with char-level span tracking
│   ├── collator.py             # Padding collator for DataLoader
│   ├── loss_masking.py         # masked_cross_entropy + auditing utilities
│   └── sft_dataset.py          # torch.utils.data.Dataset wrapper
├── utils/
│   ├── __init__.py
│   ├── optimizer.py            # AdamW factory + cosine/linear/constant scheduler
│   ├── validation.py           # Validation loop returning ValidationResult
│   ├── checkpoint_manager.py   # Save/load/prune checkpoints
│   ├── inference.py            # Autoregressive generation + sanity check runner
│   ├── logging_utils.py        # Structured training log lines
│   └── seed.py                 # Global RNG seeding
├── trainer.py                  # SFTTrainer orchestrator
├── model/
│   └── cyberslm.py             # ← Place Stage 1 architecture here (not included)
├── tokenizer/
│   └── tokenizer.model         # ← Place Stage 1 tokenizer here (not included)
├── data/
│   ├── train.jsonl             # ← Your instruction dataset
│   └── val.jsonl               # ← Optional; auto-split if absent
├── checkpoints/
│   ├── pretrained/
│   │   └── model.pt            # ← Stage 1 pretrained weights
│   ├── latest/                 # Updated every save_every_n_steps
│   ├── best/                   # Updated on each val_loss improvement
│   └── step_NNNNNNN/           # Optional versioned snapshots (keep_last_n=3)
└── tests/
    ├── test_phase1.py          # Config, loader, validator, formatter (45 tests)
    ├── test_phase2.py          # Collator, loss masking, template (37 tests)
    ├── test_phase3.py          # Optimizer, scheduler, validation (29 tests)
    └── test_phase4.py          # Checkpoint manager, inference (27 tests)
```

---

## Dataset Format

Both formats may be mixed inside the same JSONL file.

### Alpaca Format

```jsonl
{"instruction": "Explain SQL injection.", "input": "", "output": "SQL injection is ..."}
{"instruction": "Summarise the following.", "input": "Long text here.", "output": "Summary."}
```

- `instruction` — **required**, non-empty string
- `input` — **optional**; omit or set to `""` when there is no context
- `output` — **required**, non-empty string; this is what the model learns

### Conversation Format (multi-turn)

```jsonl
{
  "messages": [
    {"role": "system",    "content": "You are a cybersecurity expert."},
    {"role": "user",      "content": "What is AES?"},
    {"role": "assistant", "content": "AES (Advanced Encryption Standard) is ..."},
    {"role": "user",      "content": "What key sizes does it support?"},
    {"role": "assistant", "content": "AES supports 128, 192, and 256-bit keys."}
  ]
}
```

- `system` role is optional and placed at the start
- Multi-turn conversations are fully supported
- Every `assistant` turn contributes to the loss independently

### Validation

The `DatasetValidator` checks every sample before training:

| Check | Alpaca | Conversation |
|---|---|---|
| `instruction` non-empty | ✓ | — |
| `output` non-empty | ✓ | — |
| `input` is a string | ✓ | — |
| At least one `user` turn | — | ✓ |
| At least one `assistant` turn | — | ✓ |
| All roles in `{system, user, assistant}` | — | ✓ |
| All `content` fields non-empty | — | ✓ |

In lenient mode (default), invalid samples are dropped and training
continues.  Set `strict=True` to halt on the first violation.

---

## Prompt Template

### Alpaca — without input

```
<s>### Instruction:
{instruction}

### Response:
{output}</s>
```

### Alpaca — with input

```
<s>### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}</s>
```

### Conversation

```
<s>System: {system_content}

### User:
{user_content}

### Assistant:
{assistant_response}</s>

### User:
{next_user_content}

### Assistant:
{next_assistant_response}</s>
```

The `<s>` token is the BOS token.  The `</s>` token is the EOS token,
appended to every assistant response to teach the model when to stop.

---

## Loss Masking

Only **assistant response tokens** contribute to the cross-entropy loss.
All prompt tokens (instruction, input, system, user turns, and the
`### Response:\n` header itself) are masked with `ignore_index = -100`.

```
Token sequence:
  <s> ### Instruction: \n Explain ... \n\n ### Response: \n SQL injection is ...  </s>
  ─────────────────────────────────────────────────────  ──────────────────────────────
         MASKED (label = -100)                                  TRAINED ON
```

**Implementation**: The `PromptFormatter` tokenises the full sequence and
the prompt-only prefix independently, then masks all positions up to
`len(prompt_tokens)`.  For multi-turn conversations, the `ConversationTemplate`
tracks character-level spans for each assistant turn, which are converted to
token-level ranges to produce the label mask.

This ensures:
- Zero gradient flows through instruction/input tokens
- The model cannot "cheat" by attending to labels in a different position
- Multi-turn conversations correctly mask all user turns, not just the first

The `LossMaskAuditor` accumulates masking statistics across batches and
warns if active token ratio drops below 5% (a signal that prompts are too
long relative to responses, or that sequences are being heavily truncated).

---

## Training Workflow

### 1. Prerequisites

```bash
pip install torch sentencepiece
```

### 2. Place required files

```
model/cyberslm.py          ← Stage 1 model architecture
tokenizer/tokenizer.model  ← Stage 1 SentencePiece model
checkpoints/pretrained/model.pt  ← Stage 1 pretrained weights
data/train.jsonl           ← Your instruction dataset
```

### 3. Run training

```bash
python train.py \
    --pretrained  checkpoints/pretrained/model.pt \
    --train-data  data/train.jsonl \
    --val-data    data/val.jsonl \
    --tokenizer   tokenizer/tokenizer.model \
    --output-dir  checkpoints \
    --epochs      3 \
    --lr          2e-5 \
    --batch-size  4 \
    --accum-steps 4
```

Effective batch size = `per_device_batch_size × gradient_accumulation_steps` = **16**.

### 4. Monitor training

```
2025-07-01 10:00:00 INFO — [STEP   10/3000] loss=2.1432 ppl=8.53 lr=6.67e-07 gnorm=0.841 tok/s=8420 elapsed=12.3s
2025-07-01 10:01:23 INFO — [STEP   20/3000] loss=1.9821 ppl=7.25 lr=1.33e-06 gnorm=0.712 tok/s=9105 elapsed=95.1s
...
2025-07-01 10:15:00 INFO — [VAL   epoch=1 step=200] loss=1.7432 ppl=5.71 tok/s=11240 *** BEST ***
```

### 5. After training

The best checkpoint (lowest validation loss) is at `checkpoints/best/`.
Inference sanity checks run automatically and are printed to the log.

---

## Resume Training

Training can be interrupted at any point and resumed without loss of
progress.  Every checkpoint saves:

| File | Contents |
|---|---|
| `model.pt` | Model weights |
| `optimizer.pt` | AdamW momentum buffers |
| `scheduler.pt` | LR schedule state (current step) |
| `training_state.pt` | `{epoch, global_step, best_val_loss, val_loss}` |
| `sft_config.json` | Full configuration snapshot |
| `meta.json` | Human-readable summary |

### Resume from latest

```bash
python train.py --resume checkpoints/latest
```

### Resume from a specific step

```bash
python train.py --resume checkpoints/step_0001500
```

### Resume from config file

```bash
python train.py --config checkpoints/latest/sft_config.json --resume checkpoints/latest
```

The trainer restores the exact epoch, global step, optimizer momentum, and
LR schedule position so training continues as if it was never interrupted.

---

## Hyperparameters

### Model (frozen — must match Stage 1)

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 33.53 M |
| Hidden size | 384 |
| Layers | 12 |
| Attention heads | 6 |
| Head dimension | 64 |
| FFN size | 1024 |
| Activation | SwiGLU |
| Normalisation | RMSNorm (Pre-Norm) |
| Positional encoding | RoPE |
| Vocabulary | 32,000 |
| Context length | 4,096 |
| Weight tying | Yes |
| Bias | No |
| Dropout | 0.0 |

### SFT Training

| Parameter | Default | Notes |
|---|---|---|
| Learning rate | `2e-5` | Peak LR after warmup. ~10× lower than pretraining. |
| LR schedule | `cosine` | Cosine decay with warmup; options: `linear`, `constant` |
| Warmup ratio | `0.03` | 3% of total steps |
| Min LR ratio | `0.10` | Floor = `lr × 0.10` |
| Optimizer | AdamW | β₁=0.9, β₂=0.95, ε=1e-8 |
| Weight decay | `0.01` | Applied to non-bias/non-norm params only |
| Gradient clipping | `1.0` | Max global gradient norm |
| Gradient accumulation | `4` | Effective batch = batch_size × 4 |
| Batch size | `4` | Per-device |
| Effective batch | `16` | batch × accum |
| Epochs | `3` | Typical for SFT |
| Precision | FP32 | As specified; no mixed precision |
| Sequence length | `2048` | Data max; model supports up to 4096 |
| Seed | `42` | Global RNG seed |

### Checkpointing

| Parameter | Default |
|---|---|
| Log every N steps | 10 |
| Validate every N steps | 200 |
| Save every N steps | 500 |
| Keep last N versioned | 3 |

### Inference (post-training sanity check)

| Parameter | Default |
|---|---|
| Max new tokens | 256 |
| Temperature | 0.7 |
| Top-p (nucleus) | 0.9 |

---

## Expected Hardware

| Setup | Batch | Accum | Effective Batch | VRAM / RAM | Estimated Time (50k samples, 3 epochs) |
|---|---|---|---|---|---|
| Single A100 80GB | 16 | 1 | 16 | ~4 GB | ~1–2 hours |
| Single RTX 3090 24GB | 8 | 2 | 16 | ~6 GB | ~3–4 hours |
| Single RTX 3060 12GB | 4 | 4 | 16 | ~8 GB | ~6–8 hours |
| CPU only (dev/test) | 2 | 8 | 16 | ~4 GB RAM | Very slow; smoke-test only |

The model is 33.53 M parameters in FP32 → ~128 MB weights.  Optimizer
state (AdamW) adds ~256 MB.  Activation memory scales with batch size and
sequence length.

For sequence length 2048 and batch size 4, peak activation memory is
approximately 1–2 GB.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'model.cyberslm'`

The SFT pipeline does not include the model architecture.  Copy
`cyberslm.py` from your Stage 1 pretraining directory to `model/cyberslm.py`.

### `FileNotFoundError: Pretrained checkpoint not found`

Set `--pretrained` to the correct path.  The pipeline expects either:
- A file: `checkpoints/pretrained/model.pt`
- A directory: `checkpoints/pretrained/` (looks for `model.pt` inside)

### `All labels in this batch are IGNORE_INDEX (-100)`

Sequences are being truncated before the response begins.  Solutions:
- Increase `--max-seq-len` (up to 4096)
- Shorten your instruction/input fields
- Filter out samples where `len(instruction) + len(input) > max_seq_len * 0.8`

### Loss is NaN from step 1

- Verify the pretrained checkpoint loads correctly (check for `Pretrained base model loaded` in the log)
- Check that the tokenizer `model_path` is correct
- Try a smaller learning rate (`--lr 1e-5`)
- Verify your dataset has non-empty `output` / `assistant` fields

### Loss does not decrease

- Ensure the learning rate is not too high (> 1e-4 is too high for SFT)
- Check that loss masking is working: the `LossMaskAuditor` summary shows `active_ratio`; it should be > 5%
- Verify gradient accumulation is set correctly (effective batch should be ≥ 16)

### OOM (out of memory)

- Reduce `--batch-size` and increase `--accum-steps` to maintain effective batch
- Reduce `--max-seq-len`
- Free GPU memory used by other processes

### Resume does not restore the correct step

- Always pass `--resume checkpoints/latest` (not the model.pt path directly)
- The `training_state.pt` inside the checkpoint directory holds the step counter

### Inference outputs are incoherent

This is expected if the model has only trained for a few steps or on a
very small dataset.  After a full 3-epoch run on a quality instruction
dataset (10k+ samples), outputs should be coherent and instruction-following.

The sanity-check prompts are qualitative only.  Use `evaluate.py` and
validation loss/perplexity for quantitative assessment.

---

## Running Tests

```bash
# Phase 1 — Config, Loader, Validator, Formatter (45 tests)
python tests/test_phase1.py

# Phase 2 — Collator, Loss Masking, Template (37 tests)
python tests/test_phase2.py

# Phase 3 — Optimizer, Scheduler, Validation (29 tests)
python tests/test_phase3.py

# Phase 4 — Checkpoint Manager, Inference (27 tests)
python tests/test_phase4.py

# All phases via pytest
pytest tests/ -v
```

**Total: 138 tests across 4 phases, all passing.**

---

## Standalone Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best \
    --config     checkpoints/best/sft_config.json \
    --data       data/val.jsonl \
    --output     eval_results.json
```

Output:
```
==================================================
Evaluation Results
  Checkpoint : checkpoints/best
  Val Loss   : 1.4231
  Perplexity : 4.15
  Tok/sec    : 14200
  Tokens     : 1842910
  Batches    : 892
  Elapsed    : 129.8s
==================================================
```

---

*CyberSLM SFT Pipeline — Stage 2 of the CyberSLM project.*
