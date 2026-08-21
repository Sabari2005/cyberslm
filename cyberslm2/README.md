# CyberSLM-2

A sub-100M-parameter decoder-only language model for **cybersecurity, code, reasoning, math and
tool use**, built to be the strongest thing that fits in that budget — and honest about where that
budget stops.

Everything in this folder is written and verified. **Nothing has been trained yet**, and nothing
here will start training on your laptop: every entry point has a `--dry-run` flag, and the only
code that has actually executed so far is arithmetic and a handful of forward passes on a 5M
parameter test model.

---

## 0. Read this first: what this model can and cannot beat

You asked for a model that outperforms the current top small language models on benchmarks. I want
to be straight with you about that goal before you spend a single GPU-hour, because the honest
version of this plan is much more useful than the flattering one.

**The binding constraint is data, not architecture.**

Your corpus is **204.5M tokens** (measured from `tokenizer/cache/tokens.bin`, 665k documents). For
reference:

| Model | Params | Pretraining tokens | Ratio vs your corpus |
|---|---|---|---|
| **CyberSLM-2** | 50M | 0.2B (x4 epochs = 0.84B) | 1x |
| SmolLM2-135M | 135M | 2T | ~10,000x |
| Qwen3-0.6B | 600M | 36T | ~176,000x |
| Phi-4-mini | 3.8B | 10T | ~49,000x |

No architecture recovers a four-order-of-magnitude data gap. A model trained on 0.2B tokens will
not beat Qwen3 or SmolLM2 on MMLU, GSM8K, HumanEval or any broad benchmark, and any document that
promises otherwise is selling something. Scaling laws are not a style preference.

**So what is actually winnable?**

1. **Best-in-class at its own size.** At ~50M params trained on ~0.84B tokens, a clean modern
   recipe (GQA, Muon, WSD, document-aware packing, z-loss) should comfortably beat an equivalently
   sized model trained the way v1 was. That is a real, defensible result.
2. **Punching far above its weight on *your* domain.** This is the genuine opportunity. A 50M model
   saturated in cybersecurity text can beat a general 1-3B model on narrow security questions,
   because the larger model spent its capacity on Shakespeare and world capitals. Domain
   specialization is the one axis where small beats large, and your corpus is 60% security.
3. **Well-formed tool calling.** Syntactic reliability (valid JSON, correct tool name, proper stop
   token) is learnable from a modest amount of well-structured data. A small model can be *reliable*
   at this even when it is not *smart*.

**What it takes to genuinely compete**, in descending order of impact:

- **More pretraining data.** Getting to 5-10B tokens is the single highest-value action. Mixing in
  open corpora (FineWeb-Edu, The Stack for code, OpenWebMath) would let the 98M config train to
  completion instead of starving. Everything else is a rounding error next to this.
- **Distillation from a larger teacher.** Training against a 7B model's full output distribution
  transfers far more signal per token than a hard label does. This is how essentially every strong
  sub-100M model is actually made, and it is the only realistic route to "beats models 10x its size"
  on general benchmarks.
- **Reasoning traces.** Chain-of-thought SFT data is worth more per token than anything else for
  math and multi-step security analysis. The `<|think|>` protocol is already wired for it.

The architecture below is designed so that when you *do* have more data, nothing has to be
redesigned — you change one preset and rerun.

---

## 1. What changed from v1, and why

| Area | v1 | CyberSLM-2 | Why it matters |
|---|---|---|---|
| Attention | MHA, materialized `T x T` mask buffer | **GQA** (8q/2kv) + SDPA | 4x smaller KV cache; longer agentic traces fit |
| Stability | none | **QK-norm + z-loss** | Removes the two standard causes of loss spikes |
| Optimizer | AdamW | **Muon** (+AdamW for embeddings) | Reaches a target loss in materially fewer steps |
| Schedule | cosine | **WSD** | Resumable, and one stable phase yields several models |
| Packing | random windows, with replacement | **sequential packing + doc masking** | ~37% of tokens were never sampled per epoch before |
| Boundaries | documents bled into each other | **block-diagonal mask** | No attention across unrelated documents |
| Control tokens | literal `"</s>"` text | **real token ids** | v1 literally could not stop generating |
| Tool calling | none | **typed `<|tool_call|>` protocol** | Agentic use is a first-class trained capability |
| Reasoning | none | **`<|think|>` spans** | CoT is trainable and maskable at inference |
| Verification | none | **`verify_architecture.py`** | Param math and masking are machine-checked |

---

## 2. Architecture

Decoder-only pre-norm transformer. Three verified presets:

| Preset | Params | d_model | Layers | Heads (q/kv) | FFN | KV cache @2k |
|---|---|---|---|---|---|---|
| `flagship-98m` | 98,324,736 | 768 | 12 | 12 / 2 | 2048 | 12.6 MB |
| `base-50m` **(default)** | 50,608,128 | 512 | 12 | 8 / 2 | 1408 | 12.6 MB |
| `tiny-5m` | 4,949,376 | 128 | 4 | 4 / 1 | 384 | 0.1 MB |

`base-50m` is the default **because it is the one your corpus can actually train.** At 20
tokens/param it wants ~1.01B tokens, which is 4.9 passes over your data — right at the edge of where
repeating data still pays. `flagship-98m` would need 9.6 passes and would overfit; use it once you
have ~2B tokens.

Verify any of this yourself, instantly, with no GPU:

```bash
python -m cyberslm2.scripts.verify_architecture            # arithmetic only
python -m cyberslm2.scripts.verify_architecture --with-torch   # + CPU forward passes
```

The `--with-torch` run empirically confirms: analytic parameter count equals the real one,
attention is causal, packed documents cannot see each other, RoPE scores depend only on relative
distance, and KV-cached decoding matches full recomputation exactly.

### Component rationale

**Grouped-Query Attention (8 query heads, 2 KV heads).** The KV cache — not the weights — dominates
inference memory, and it scales with the number of KV heads. Sharing 4 query heads per KV head cuts
the cache 4x for a small, well-documented quality cost. For an agentic model that must hold long
tool-call traces in context, this trade is clearly correct.

**QK-norm.** RMSNorm applied to queries and keys before the dot product. Attention logits otherwise
drift to large magnitudes at high learning rates, which is the most common origin of loss spikes in
small models. Costs `2 * head_dim` parameters per layer.

**z-loss (1e-4).** Cross-entropy is invariant to a constant shift in the logits, so nothing pins the
log-partition function. Penalizing `logsumexp(logits)^2` pins it near zero without touching the
relative logits that carry the prediction.

**SwiGLU at (8/3)·d.** Three matrices instead of two, so the width is reduced to hold parameters
constant. The multiplicative gate is worth roughly 1-2% loss at equal size.

**Tied embeddings.** Saves 16.8M parameters in `base-50m` — a third of the budget — and regularizes
by forcing the input and output views of a token to agree.

**Depth-scaled init.** Residual-writing projections are scaled by `1/sqrt(2*n_layers)`. Each of the
`2N` sublayers adds independent variance to the residual stream; without this the stream's variance
grows linearly with depth and deep models start off badly conditioned.

---

## 3. Token protocol

Everything is a **real token id**. This is the direct lesson from v1, where `"</s>"` was written as
characters, got shredded into ordinary pieces, and the model never learned to stop.

| id | token | role |
|---|---|---|
| 0-3 | `<pad>` `<unk>` `<bos>` `<eos>` | SentencePiece control slots |
| 4-7 | `<\|system\|>` `<\|user\|>` `<\|assistant\|>` `<\|end\|>` | chat turns |
| 8-9 | `<\|think\|>` `<\|/think\|>` | private reasoning span |
| 10-13 | `<\|tool_list\|>` `<\|tool_call\|>` `<\|tool_result\|>` `<\|/tool\|>` | agentic protocol |
| 14-15 | `<\|code\|>` `<\|/code\|>` | executable code |
| 16-35 | `<\|reserved_*\|>` | headroom, so adding a token later never shifts ids |

```
<bos><|user|>Scan 10.0.0.5 for open services<|end|>
<|assistant|><|think|>Port scan first, then version detection.<|/think|>
<|tool_call|>{"name": "nmap_scan", "arguments": {"target": "10.0.0.5", "flags": "-sV"}}<|/tool|><|end|>
<|tool_result|>{"ok": true, "output": "22/tcp open ssh OpenSSH 8.9"}<|/tool|>
<|assistant|>The host runs OpenSSH 8.9 on port 22.<|end|><eos>
```

**Loss is computed only on assistant tokens.** Tool *results* are always masked out — training on
them teaches the model to invent tool output, which is the most damaging possible failure for an
agent.

---

## 4. Training recipe

### Stage 0 — retokenize (required)

The v2 tokenizer adds the control tokens and moves to a 32,768 vocab, so the corpus must be
retokenized. This also picks up the `--add-eos` document-boundary fix.

```bash
python -m cyberslm2.scripts.train_tokenizer_v2 --dry-run     # inspect the id layout
python -m cyberslm2.scripts.train_tokenizer_v2 \
    --input tokenizer/Final.jsonl --output-dir tokenizer/v2
```

`split_digits=True` is not optional if you want arithmetic: it forces `4096` to tokenize as
`4/0/9/6`, because a model cannot do column-wise math on a token that hides the columns.
`byte_fallback=True` guarantees hex dumps, base64 and shellcode are always encodable.

### Stage 1 — pretrain

```bash
python -m cyberslm2.scripts.pretrain --dry-run               # safe anywhere
python -m cyberslm2.scripts.pretrain --preset base-50m       # needs a GPU
```

Muon at 3e-3 for hidden matrices, AdamW at 3e-4 for embeddings and norms, WSD schedule,
262k tokens/step, 3,200 steps = 839M tokens = 4.1 epochs.

### Stage 2 — SFT

```bash
python -m cyberslm2.scripts.sft --dry-run
python -m cyberslm2.scripts.sft --pretrained cyberslm2/checkpoints/pretrain/best.pt
```

AdamW at 2e-5, 2,000 steps, light dropout (0.05) since SFT sets are small enough to memorize.

### Compute estimate

`base-50m` at 839M tokens is roughly **2.6 x 10^17 FLOPs**:

| Hardware | Realistic time |
|---|---|
| A100 80GB | ~35 min |
| RTX 4090 | ~1 hour |
| **T4 (free Colab)** | **~4.5 hours** |

This fits in free Colab. That is the point of sizing the model to the data rather than to the
parameter ceiling.

**Your laptop has no GPU — do not run stages 1 or 2 locally.** Use `--dry-run` to validate configs
here, then run the real thing on Colab or a rented GPU.

---

## 5. Evaluation

`eval/harness.py` scores two ways. **Loglikelihood** (multiple choice) picks the
highest length-normalized log-probability among candidates — no generation, so it is cheap and
stable. **Generative** actually decodes and checks exact-match, substring, or *tool-call validity*,
where a call counts only if it parses as JSON, carries a `name`, and closes properly. Partial credit
would hide exactly the failures that break a real agent loop.

Drop task files into a directory as `mc_*.jsonl` or `gen_*.jsonl` and `run_suite` picks the scorer
by prefix.

**Build the security eval set before you train.** A held-out bank of a few hundred questions from
your own domain is worth more than any public leaderboard here, because it measures the thing this
model is actually for.

---

## 6. Layout

```
cyberslm2/
├── configs/presets.py       ModelConfig + TrainConfig + the 3 presets
├── model/
│   ├── norm.py              RMSNorm
│   ├── rope.py              RoPE with long-context scaling
│   ├── ffn.py               SwiGLU
│   ├── attention.py         GQA + QK-norm + SDPA + KV cache
│   ├── block.py             pre-norm decoder block
│   └── transformer.py       full model, init, generation
├── data/
│   ├── special_tokens.py    the id protocol + validation
│   ├── packing.py           doc-aware block-diagonal masking
│   └── datasets.py          packed pretrain + SFT datasets
├── training/
│   ├── optim.py             Muon (Newton-Schulz) + AdamW
│   ├── schedule.py          WSD + cosine
│   ├── loss.py              CE + z-loss
│   └── trainer.py           the loop
├── eval/harness.py          loglikelihood + generative scoring
├── scripts/
│   ├── verify_architecture.py   machine-checked design
│   ├── train_tokenizer_v2.py
│   ├── pretrain.py
│   └── sft.py
├── RESEARCH_PROMPT.md       the research brief driving this design
└── README.md
```

---

## 7. Ethics

Security content is dual-use. This model is scoped to **defensive** work: analysis, detection,
hardening, secure coding, incident response and education. Do not train it to generate working
exploits against systems you do not own, and keep the SFT data framed accordingly. A model that
refuses nothing is not more capable, only more liable.

---

## 8. Status

- [x] Architecture designed, parameter math verified under 100M
- [x] Causality, document masking, RoPE and KV-cache correctness empirically checked
- [x] Full training / SFT / eval code written; all 26 modules import cleanly
- [ ] Retokenize with the v2 tokenizer
- [ ] Build the held-out security eval set
- [ ] Pretrain (GPU required — not on this laptop)
- [ ] SFT, then evaluate
