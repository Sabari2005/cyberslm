# CyberSLM

A **cybersecurity-focused decoder-only Small Language Model** built completely
from scratch in PyTorch.  Every component is original — no GPT, LLaMA, Gemma,
Mistral, or other reference implementation was used.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Mathematical Components](#mathematical-components)
3. [Directory Structure](#directory-structure)
4. [Hyperparameters](#hyperparameters)
5. [Training Workflow](#training-workflow)
6. [Checkpoint Format](#checkpoint-format)
7. [Resume Training](#resume-training)
8. [Hardware Recommendations](#hardware-recommendations)
9. [Expected Memory Usage](#expected-memory-usage)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
Input token IDs  (B, T)
        │
        ▼
Token Embedding  (vocab_size=32 000 → hidden_dim=384)
        │
        ▼
┌──────────────────────────────────────────┐
│  Decoder Block × 12                      │
│                                          │
│  ┌──────────┐   ┌──────────────────────┐ │
│  │ RMSNorm  │──▶│ Multi-Head Attention │ │
│  └──────────┘   │  (6 heads, 64 dim)   │ │
│       ⊕◀────────│  + RoPE + CausalMask │ │
│       │         └──────────────────────┘ │
│  ┌──────────┐   ┌──────────────────────┐ │
│  │ RMSNorm  │──▶│   SwiGLU FFN         │ │
│  └──────────┘   │   (1024 inner dim)   │ │
│       ⊕◀────────└──────────────────────┘ │
└──────────────────────────────────────────┘
        │
        ▼
Final RMSNorm
        │
        ▼
LM Head  (384 → 32 000)  [weight-tied to embedding]
        │
        ▼
Logits  (B, T, vocab_size)
```

**Design principles:**
- **Pre-Norm** residual connections for training stability
- **RoPE** positional encoding for relative-position generalisation
- **SwiGLU** activations for improved feed-forward quality
- **RMSNorm** (no mean subtraction) for efficiency and stability
- **Weight tying** between embedding and LM head (saves ~12.3 M parameters)
- **No biases** on any linear layer (modern practice)
- **No dropout** (pure pretraining from scratch)

---

## Mathematical Components

### RMSNorm

Given input **x** ∈ ℝ^d:

```
RMS(x) = sqrt( (1/d) Σᵢ xᵢ² + ε )
RMSNorm(x) = (x / RMS(x)) ⊙ γ
```

Where γ ∈ ℝ^d is a learned per-channel scale (initialised to 1).
ε = 1e-6 prevents division by zero.

**Key property:** Scale-invariant for well-conditioned inputs —
`RMSNorm(αx) = RMSNorm(x)` when ε ≪ mean(xᵢ²).  
**Note:** This invariance breaks for very small inputs (α → 0) because
ε becomes comparable to mean(x²). This is intentional and correct.

**Stability:** RMS is computed in float32 regardless of input dtype.

---

### Rotary Position Embedding (RoPE)

For query **q** at position m with head dimension d_h:

1. Define frequencies: `θᵢ = 1 / 10000^(2i / d_h)` for i ∈ {0, …, d_h/2 − 1}

2. Apply 2-D rotation to each consecutive pair (q_{2i}, q_{2i+1}):
```
q'_{2i}   = q_{2i}   · cos(m·θᵢ) − q_{2i+1} · sin(m·θᵢ)
q'_{2i+1} = q_{2i}   · sin(m·θᵢ) + q_{2i+1} · cos(m·θᵢ)
```

**Relative position property:** The dot product `⟨R(m)q, R(n)k⟩` depends
only on the offset (m − n), not absolute positions.  This gives the attention
mechanism translational equivariance without learnable position embeddings.

**Implementation:** Frequencies precomputed in float64, cached as float32
buffers. Applied via `x_rot = x·cos + rotate_half(x)·sin`.

---

### Causal Mask

Upper-triangular additive mask M ∈ {0, −∞}^{T×T}:

```
M[i,j] = 0      if j ≤ i   (attend to past and present)
M[i,j] = −∞    if j > i   (block future positions)
```

Added to raw attention scores before softmax, so future positions contribute
exactly 0 after `exp(−∞) = 0`.

---

### Multi-Head Self Attention

```
Q = XWq,  K = XWk,  V = XWv          ∈ ℝ^{B×T×d}
Qₕ, Kₕ = RoPE(Qₕ), RoPE(Kₕ)         per head
Aₕ = softmax((Qₕ Kₕᵀ)/√d_h + M) Vₕ
output = concat(A₁,…,Aₕ) Wo
```

- Scale: `1/√d_h = 1/√64 = 0.125` prevents pre-softmax logit explosion
- All projection matrices: no bias
- Softmax computed in float32 for stability

**Parameter count per layer:** 4 × (384 × 384) = 589,824

---

### SwiGLU Feed-Forward Network

```
gate(x)  = x Wgate           ∈ ℝ^{ffn_dim}
val(x)   = x Wval            ∈ ℝ^{ffn_dim}
hidden   = swish(gate(x)) ⊙ val(x)
output   = hidden Wout        ∈ ℝ^{hidden_dim}
```

Where `swish(z) = z · σ(z) = z / (1 + e^{-z})`.

The gating mechanism gives the network a multiplicative path to
selectively pass or suppress information at each position and dimension.

**Parameter count per layer:** 3 × (384 × 1024) = 1,179,648

---

### Weight Tying

The output (unembedding) matrix `Wo ∈ ℝ^{vocab×hidden}` shares storage with
the token embedding matrix `E ∈ ℝ^{vocab×hidden}`:

```python
lm_head.weight = embedding.weight   # same Python object
```

**Justification:** Both matrices operate in the same semantic space —
the embedding maps token ID → semantic vector, and the unembedding maps
semantic vector → token logits.  Tying them improves generalisation on
small models and saves 12,288,000 parameters (37% of total).

---

### Parameter Initialisation

| Component | Distribution | Notes |
|-----------|-------------|-------|
| Embeddings | N(0, 0.02) | Standard for transformer LMs |
| Linear weights | N(0, 0.02) | All non-output projections |
| o_proj, out_proj | N(0, 0.02/√(2L)) | Scaled to control residual variance at init |
| RMSNorm γ | 1.0 | Identity at initialisation |

The output projection scaling `1/√(2L)` (where L=12) ensures the residual
stream variance at layer n does not grow as O(n) at initialisation.

---

## Directory Structure

```
cyberslm/
├── __init__.py
│
├── model/
│   ├── __init__.py          # Public API
│   ├── config.py            # CyberSLMConfig (frozen dataclass)
│   ├── norm.py              # RMSNorm
│   ├── rope.py              # RotaryPositionEmbedding
│   ├── mask.py              # CausalMask
│   ├── attention.py         # MultiHeadSelfAttention
│   ├── ffn.py               # SwiGLUFeedForward
│   ├── block.py             # DecoderBlock (Pre-Norm)
│   └── model.py             # CyberSLM, build_model, model_summary
│
├── training/
│   ├── __init__.py          # Public API
│   ├── config.py            # TrainingConfig
│   ├── scheduler.py         # CosineWarmupScheduler
│   ├── checkpoint.py        # CheckpointManager
│   ├── metrics.py           # StepMetrics, memory helpers
│   └── trainer.py           # Trainer, build_optimizer, compute_lm_loss
│
└── scripts/
    ├── verify_phase1.py     # Config · RMSNorm · RoPE tests
    ├── verify_phase2.py     # CausalMask · Attention · FFN tests
    ├── verify_phase3.py     # Full model tests
    ├── verify_phase4.py     # Training engine tests
    └── verify_all.py        # Master suite + integration tests
```

---

## Hyperparameters

### Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | Decoder-only Transformer | |
| Vocabulary | 32,000 | SentencePiece BPE |
| Max context | 4,096 tokens | |
| Hidden dim (d) | 384 | |
| Decoder layers | 12 | |
| Attention heads | 6 | |
| Head dim | 64 | = 384 / 6 |
| FFN inner dim | 1,024 | |
| Activation | SwiGLU | |
| Normalisation | RMSNorm, ε=1e-6 | Pre-Norm |
| Position encoding | RoPE, base=10,000 | |
| Weight tying | Enabled | embedding ↔ LM head |
| Linear bias | Disabled | |
| Dropout | 0.0 | |
| **Total parameters** | **33,531,264** | **~33.53 M** |

### Training (defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | |
| β₁, β₂ | 0.9, 0.95 | |
| ε | 1e-8 | |
| Peak LR | 3e-4 | |
| Min LR | 3e-5 | 1/10 of peak |
| LR schedule | Linear warmup + cosine decay | |
| Warmup steps | 400 | 10% of total |
| Max steps | 4,000 | ~5 epochs on 203M tokens |
| Weight decay | 0.1 | matrices only; 1-D params exempt |
| Gradient clip | 1.0 | global norm |
| Batch size | 4 sequences/GPU | |
| Grad accum steps | 16 | effective batch = 64 |
| Effective batch | 64 seqs × 4,096 tok = 262,144 tokens | |
| Sequence length | 4,096 | |

---

## Training Workflow

### 1. Prerequisites

```bash
pip install torch numpy
```

### 2. Verify the implementation

```bash
# Run all checks (takes ~2 min on CPU)
python -m cyberslm.scripts.verify_all

# Or run individual phases:
python -m cyberslm.scripts.verify_phase1
python -m cyberslm.scripts.verify_phase2
python -m cyberslm.scripts.verify_phase3
python -m cyberslm.scripts.verify_phase4
```

### 3. Inspect the model

```python
from cyberslm.model.model import build_model, model_summary

model = build_model()
print(model_summary(model))
```

### 4. Training script (single GPU)

```python
import logging
import torch
from torch.utils.data import DataLoader

from cyberslm.model.config import default_config
from cyberslm.training.config import TrainingConfig
from cyberslm.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Your memory-mapped DataLoader (already implemented).
train_loader = DataLoader(...)   # batches: (batch_size, seq_len) long
val_loader   = DataLoader(...)

model_config = default_config()
train_config = TrainingConfig(
    train_data_path="data/train.bin",
    val_data_path="data/val.bin",
    batch_size=4,
    grad_accum_steps=16,
    max_steps=4_000,
    warmup_steps=400,
    checkpoint_dir="checkpoints",
)

trainer = Trainer(
    model_config=model_config,
    train_config=train_config,
    train_loader=train_loader,
    val_loader=val_loader,
)
trainer.train()
```

### 5. Multi-GPU training (DDP)

```python
train_config = TrainingConfig(distributed=True, ...)
```

Launch with:
```bash
torchrun --nproc_per_node=4 your_train_script.py
```

### 6. torch.compile (optional, PyTorch 2.0+)

```python
train_config = TrainingConfig(compile_model=True, ...)
```

Typical speedup on A100: ~20–40% after compilation warmup.

---

## Checkpoint Format

Each checkpoint is a single `.pt` file with:

```python
{
    "step":             int,          # optimizer step number
    "model_state":      OrderedDict,  # model.state_dict()
    "optimizer_state":  dict,         # optimizer.state_dict()
    "scheduler_state":  dict,         # {step, peak_lr, min_lr, ...}
    "train_loss":       float,        # last training loss
    "val_loss":         float | None, # validation loss (if available)
    "config":           dict,         # CyberSLMConfig as plain dict
    "train_config":     dict,         # TrainingConfig as plain dict
    "rng_state":        dict,         # torch/numpy/python RNG states
}
```

### Files created during training

```
checkpoints/
├── ckpt_step_0000200.pt    # periodic (rotated, keeps last 3)
├── ckpt_step_0000400.pt
├── ckpt_step_0000600.pt
├── best.pt                  # best validation loss (never rotated)
└── latest.txt               # path of the most recent checkpoint
```

---

## Resume Training

```python
trainer = Trainer(
    model_config=model_config,
    train_config=train_config,
    train_loader=train_loader,
    val_loader=val_loader,
    resume_from="checkpoints/ckpt_step_0002000.pt",
)
trainer.train()
```

Or resume from the latest checkpoint automatically:

```python
from cyberslm.training.checkpoint import CheckpointManager

ckpt_mgr = CheckpointManager("checkpoints")
latest   = ckpt_mgr.latest_checkpoint()

trainer = Trainer(..., resume_from=latest)
```

**What is restored:** model weights, optimizer state (momentum/variance),
scheduler step, and RNG states (for deterministic data ordering if your
DataLoader is seeded consistently).

---

## Hardware Recommendations

### Minimum (testing / verification)

- CPU only, 8 GB RAM
- Slow but functional for correctness checks

### Recommended (training)

| Setup | Batch/GPU | Grad Accum | Effective Batch | Est. Time (203M tok) |
|-------|-----------|------------|-----------------|----------------------|
| 1× RTX 3090 (24 GB) | 4 | 16 | 64 seqs | ~8–10 hours |
| 1× A100-40GB | 8 | 8 | 64 seqs | ~4–5 hours |
| 2× A100-40GB | 8 | 4 | 64 seqs | ~2–3 hours |
| 4× A100-80GB | 16 | 2 | 128 seqs | ~1–2 hours |

### Sequence length impact

Attention is O(T²) in both time and memory.  Reducing sequence length
(`seq_len=2048`) roughly halves GPU memory and quadruples attention speed,
at the cost of less context per training step.

---

## Expected Memory Usage

For one GPU at FP32 with default settings (batch=4, seq=4096):

| Component | Memory |
|-----------|--------|
| Model parameters | ~128 MB |
| Optimizer state (AdamW m + v) | ~256 MB |
| Activations (batch=4, seq=4096) | ~1.5–2.5 GB |
| Attention score matrix (6 heads) | ~384 MB |
| Gradient buffers | ~128 MB |
| **Total (approx.)** | **~2.5–3.5 GB** |

> Gradient accumulation allows trading GPU memory for larger effective batch
> size without increasing peak activation memory (each micro-batch is processed
> and discarded before the next).

---

## Troubleshooting

### NaN loss at training start

**Cause:** Learning rate too high, or bad initialisation.  
**Fix:** Confirm warmup is active (`warmup_steps > 0`). Try `learning_rate=1e-4`.

### Loss not decreasing

**Cause:** Learning rate too low, or gradient clipping too aggressive.  
**Fix:** Check that `scheduler.current_step` is advancing. Check `grad_norm`
in logs — if it's always exactly `grad_clip`, reduce clip or increase LR.

### CUDA out of memory

**Cause:** Batch × sequence length too large.  
**Fix options:**
1. Reduce `batch_size` (e.g., 4 → 2) and increase `grad_accum_steps` to compensate.
2. Reduce `seq_len` (e.g., 4096 → 2048).

### Checkpoint resume produces different loss

**Cause:** DataLoader state is not restored (only model/optimizer/scheduler
and RNG are saved).  
**Fix:** Ensure your DataLoader uses the same seed and starting offset
on resume, or accept a small discontinuity in the first few steps.

### Verification fails: `head_dim must equal hidden_dim // num_heads`

**Cause:** Modified `CyberSLMConfig` with inconsistent values.  
**Fix:** Always set `head_dim = hidden_dim // num_heads` explicitly, or use
`default_config()` which has this correct by construction.

### Very slow training on CPU

**Cause:** Attention is O(T²·d) per layer.  At T=4096, seq length dominates.  
**Fix:** Use `seq_len=512` or lower for CPU debugging.  GPU is required for
production training.

### `torch.compile` errors

**Cause:** Dynamic shapes (variable sequence lengths).  
**Fix:** Ensure all batches have the same `seq_len` (your DataLoader should
guarantee this).  Alternatively, set `compile_model=False`.

### `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Model and input on different devices.  
**Fix:** `batch = batch.to(device)` before every forward pass.  The `Trainer`
handles this automatically; check your custom training script.
