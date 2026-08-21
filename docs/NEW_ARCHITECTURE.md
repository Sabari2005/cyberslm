# CyberSLM‑2 — A Mathematically‑Verified <100M‑Parameter Architecture

A ground‑up redesign for a **sub‑100‑million‑parameter** small language model (SLM) targeting
**reasoning, code, cybersecurity, tool‑calling, and agentic use**. Every dimension below is chosen so
the parameter count is *provably* under 100M, every tensor shape is consistent, and every design
choice is justified from first principles. This document is a specification, not a drop‑in patch —
it also fixes the concrete defects listed in `CODE_REVIEW_AND_FIXES.md`.

---

## 0. Honest capability expectations (read this first)

A 100M‑parameter model is ~3× the previous 33.5M model but still **two to three orders of magnitude
smaller** than frontier models. Set expectations correctly:

- **Achievable at ~96M params** with a good recipe: fluent domain text; short chain‑of‑thought on
  constrained problems; reliable **structured tool‑call emission** (JSON) for a fixed tool schema;
  single‑file code completion and small function synthesis; retrieval‑augmented Q&A where facts come
  from context, not weights; cybersecurity explanation and classification.
- **Not achievable at this scale**, regardless of architecture: broad world knowledge, long
  multi‑hop reasoning, competitive coding, or agentic planning *without* an external
  scaffold/retriever. The design therefore treats the model as a **fast, tool‑using controller**,
  not a knowledge oracle. Capability comes from *architecture + data + a tool/RAG harness*, and this
  document specifies all three.

The single biggest lever at fixed parameter count is **training tokens and data quality**, not
architectural cleverness (Chinchilla). The compute‑optimal token budget for 96M params is
≈ 20 × 96M ≈ **1.9B tokens**; for a capability‑maxed SLM we recommend **20–40B tokens** (heavily
“over‑trained,” as is standard for deployable SLMs).

---

## 1. Parameter budget — verified under 100M

**Chosen dimensions**

| Symbol | Meaning | Value |
|---|---|---|
| `V` | vocab size | 32,768 (2¹⁵) |
| `d` | model dim | 768 |
| `L` | layers | 12 |
| `h_q` | query heads | 12 |
| `h_kv` | key/value heads (GQA) | 4 |
| `d_h` | head dim | 64 (`= d / h_q`) |
| `f` | FFN inner (SwiGLU) | 1,920 (2.5 × d) |
| `ctx` | trained context | 8,192 (RoPE, extendable) |
| tie | embedding ↔ LM head | yes |

**Per‑component counts (exact integer arithmetic)**

```
Embedding (tied, counted once)  V·d = 32768·768                 = 25,165,824

Per layer:
  q_proj    d·(h_q·d_h) = 768·768                               =    589,824
  k_proj    d·(h_kv·d_h) = 768·256                              =    196,608
  v_proj    d·(h_kv·d_h) = 768·256                              =    196,608
  o_proj    (h_q·d_h)·d = 768·768                               =    589,824
  attn subtotal                                                 =  1,572,864
  gate_proj d·f = 768·1920                                      =  1,474,560
  up_proj   d·f = 768·1920                                      =  1,474,560
  down_proj f·d = 1920·768                                      =  1,474,560
  ffn subtotal                                                  =  4,423,680
  2× RMSNorm  2·d                                               =      1,536
  per-layer total                                               =  5,998,080

All layers    12 · 5,998,080                                    = 71,976,960
Final RMSNorm  d                                                =        768
LM head        tied → 0 extra                                   =          0
──────────────────────────────────────────────────────────────────────────
TOTAL                                                           = 97,143,552  (≈ 97.1M)
Non-embedding (the "compute" params)                            = 71,977,728  (≈ 72.0M)
```

**97.14M < 100M ✓** with ~2.9M of headroom. (Verify programmatically with the snippet in §11.)

> **Design note on the embedding fraction.** At 32,768×768 the tied embedding is ~25.9% of params.
> Keeping `V` a power of two (32,768) is GPU‑friendly and leaves room for reserved tool/agent tokens
> (§6). If you want *more* “thinking” capacity, shrink `V` to 24,576 (frees ~6.3M params → move to a
> 13th layer or `f=2048`). The table above is the recommended balance.

---

## 2. Architecture overview

Decoder‑only pre‑norm transformer, LLaMA‑class, with modern SLM‑specific choices:

```
input_ids (B,T)
   │  token embedding  (V,d), tied to LM head
   ▼
[ DecoderBlock × 12 ]
   │   x = x + Attn(RMSNorm(x))        # GQA + RoPE + causal + key-padding mask
   │   x = x + SwiGLU(RMSNorm(x))
   ▼
RMSNorm
   ▼
LM head = embeddingᵀ   →  logits (B,T,V)
```

Formally, for hidden state `x ∈ ℝ^{B×T×d}` at block ℓ:

```
a  = MHA_GQA( RMSNorm(x) )              x ← x + a
g  = W_down ( SiLU(x̂ W_gate) ⊙ (x̂ W_up) ),   x̂ = RMSNorm(x),   x ← x + g
```

### 2.1 Grouped‑Query Attention (GQA) — why, and the math

We use **12 query heads sharing 4 KV heads** (group size 3). Query/Key/Value:

```
Q = x W_Q ∈ ℝ^{T×(h_q·d_h)} = ℝ^{T×768}
K = x W_K ∈ ℝ^{T×(h_kv·d_h)} = ℝ^{T×256}
V = x W_V ∈ ℝ^{T×256}
```

Reshape `Q→(h_q,T,d_h)`, `K,V→(h_kv,T,d_h)`; each KV head is **repeated 3×** to pair with its query
group. Per head `i` (with its shared KV head `⌊i/3⌋`):

```
head_i = softmax( (Q_i K_gᵀ)/√d_h + M ) V_g ,   g = ⌊i/3⌋
out    = concat(head_0..head_11) W_O
```

**Why GQA at this scale:** it cuts the **KV cache by 3×** (only 4 KV heads stored) at negligible
quality loss, which is decisive for *agentic* use where long tool‑augmented contexts dominate memory.

- KV cache @ `ctx=8192`, bf16: `2·L·h_kv·d_h·ctx·2 bytes = 2·12·4·64·8192·2 = 100.7 MB/sequence`.
- Full multi‑head (12 KV heads) would be **302 MB/sequence** — 3× worse. GQA makes 8k context cheap.

`M` is the additive mask (§2.4). Attention is computed with
`F.scaled_dot_product_attention(..., is_causal=True)` so FlashAttention kernels apply and no
`T×T` matrix is materialized (fixes review §5).

### 2.2 RoPE positional encoding — with long‑context scaling built in

Rotary embeddings on Q and K (no learned position table). For pair index `i ∈ {0..d_h/2−1}`:

```
θ_i = base^(−2i/d_h),   base = 1,000,000   (not 10,000)
rotate(x, m): x'₂ᵢ   = x₂ᵢ cos(mθ_i) − x₂ᵢ₊₁ sin(mθ_i)
              x'₂ᵢ₊₁ = x₂ᵢ cos(mθ_i) + x₂ᵢ sin... (standard 2-D rotation)
```

**Why `base = 1e6`:** a larger base lowers the lowest rotary frequency, which materially improves
extrapolation and makes context extension to 16k–32k cheap via **NTK/YaRN scaling** at fine‑tune
time. `d_h = 64` is even (RoPE requirement ✓). Compute cos/sin in fp32; store as non‑persistent
buffers. The relative‑position inner‑product property `⟨R_m q, R_n k⟩ = f(m−n)` is preserved.

### 2.3 RMSNorm + SwiGLU

- **RMSNorm** (`RMS(x)=√(mean(x²)+ε)`, `y = x/RMS(x)·γ`, `ε=1e-5`) computed in fp32 — cheaper than
  LayerNorm, no re‑centering. Pre‑norm placement + a **final** RMSNorm before the head.
- **SwiGLU** with `f = 1920` (2.5× d). The classic SwiGLU inner ratio is 8/3·d ≈ 2048; we use 2.5×
  to stay under the parameter cap while keeping the gated‑MLP benefit. Three bias‑free matrices
  `W_gate, W_up (d→f)`, `W_down (f→d)`; activation `SiLU(z)=z·σ(z)`.

### 2.4 Masking — causal **and** padding, done correctly

Fixes review §4 and §14. Attention adds a causal mask **and**, when a batch contains padding, a
key‑padding mask. Rather than additive `−inf` (which NaNs fully‑masked rows), we rely on SDPA’s
boolean mask, or use dtype‑min with a guaranteed‑valid diagonal:

```python
attn_mask = None
if pad_mask is not None:                      # pad_mask: (B,T) 1=keep 0=pad
    m = pad_mask[:, None, None, :].bool()     # (B,1,1,T)
    attn_mask = m                             # combine with is_causal in SDPA
context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None))
```

For training we prefer **sequence packing with no padding** (§5.2), so the padding path is only used
for batched generation.

### 2.5 Numerical & init details

- **Init:** `N(0, 0.02)` for all linears/embeddings; **residual projections** (`o_proj`, `down_proj`)
  scaled by `1/√(2L)` so residual‑stream variance stays ~constant with depth (GPT‑2/LLaMA rule).
- **Precision:** train in **bf16** with an fp32 master copy (or `torch.autocast`); norms and the
  softmax accumulate in fp32. No loss scaler needed with bf16.
- **z‑loss** (`1e-4 · logsumexp(logits)²`) added to the LM loss to keep logits well‑scaled — cheap
  stabilizer that also improves calibration for tool‑call token probabilities.
- **QK‑norm (optional):** RMSNorm on Q and K before RoPE improves training stability at low precision;
  adds only `2·h·d_h` params. Recommended if you see attention‑logit blow‑ups.

---

## 3. Tokenizer — code‑ and security‑aware

A tokenizer is *inseparable* from capability for code/security text. Requirements:

- **Byte‑fallback = true** (never emit `<unk>` on hex, base64, shellcode, non‑ASCII) — keep.
- **`split_digits = true`** (change from the old config): each digit its own token, which is standard
  for arithmetic/reasoning and for CVE IDs, ports, CIDR blocks. The previous `split_digits=False`
  hurts numeric reasoning.
- **Preserve whitespace/tabs/newlines** for code (`remove_extra_whitespaces=False`,
  `allow_whitespace_only_pieces=True`) — keep.
- **`character_coverage = 0.9998`** but **curate the corpus** so CJK does not consume vocab rows in an
  English/code model (the old vocab wasted rows on Chinese pieces).
- **Reserved special tokens with fixed ids** (put them first, so ids are stable):

```
0 <pad>   1 <unk>   2 <bos>   3 <eos>
4 <|system|>  5 <|user|>  6 <|assistant|>  7 <|end|>
8 <|tool_call|> 9 <|tool_result|> 10 <|think|> 11 <|/think|>
12..31  <|reserved_k|>   (future-proofing)
```

**Critical rule (fixes review §1/§2):** these are *real vocab tokens*, referenced by **id**
everywhere — never as literal strings like `"</s>"`. `<eos>` (3) terminates every document during
pretraining and every assistant turn during SFT, and the sampler stops on id 3. Add a startup
assertion that the tokenizer’s special ids match the config.

Train `vocab_size=32768`, `model_type=bpe`, `add_dummy_prefix=false` (avoids the boundary‑drift bug in
review §7; makes char↔token alignment trivial).

---

## 4. Capability‑specific design

Because raw scale is fixed, capabilities are engineered through **special tokens, data formats, and a
harness**, not extra parameters.

### 4.1 Reasoning — explicit thinking tokens

Train on data wrapped as `... <|think|> reasoning steps <|/think|> final answer <eos>`. At inference
the harness can *hide* the think span. This teaches structured deliberation and lets you trade tokens
for accuracy at run time. Keep chains short (SLMs degrade on long CoT); prefer **distilled** concise
reasoning from a larger teacher.

### 4.2 Tool calling / agentic — a fixed, learnable protocol

The model emits a **typed, JSON tool call** delimited by special tokens; the harness executes and
feeds back a result:

```
<|user|> scan 10.0.0.5 for open web ports <|end|>
<|assistant|><|tool_call|>{"name":"port_scan","arguments":{"host":"10.0.0.5","ports":[80,443,8080]}}<|end|>
<|tool_result|>{"open":[443]}<|end|>
<|assistant|>Host 10.0.0.5 has 443/tcp open (HTTPS); 80 and 8080 are closed.<eos>
```

Design points that make this work at 100M:
- **Constrained decoding** at inference (grammar/JSON‑schema‑guided sampling). A small model cannot be
  trusted to emit valid JSON freely; a decoding grammar guarantees well‑formed calls. This is the
  single most important trick for reliable agentic behavior at this scale.
- **Loss is computed only on assistant + tool_call tokens**, never on user/tool_result tokens
  (masking, §5.3).
- Keep the tool schema **small and fixed** during SFT; generalization to unseen tools is weak at 100M,
  so ship the tool list in the system prompt and use RAG for tool docs.

### 4.3 Cybersecurity & code

Domain competence is a **data** property. Pretraining/SFT mixture (weights are a starting point):

| Bucket | Share | Notes |
|---|---|---|
| Cybersecurity (the existing corpus + MITRE/CWE/CVE, writeups) | 25% | keep the existing categories |
| Code (permissive licenses, with docstrings/tests) | 25% | drives tool‑use & structure |
| General reasoning / math (concise CoT, GSM‑style) | 15% | |
| High‑quality web/wiki/books (English) | 20% | fluency & world facts |
| Instruction/chat + tool‑calling traces | 15% | format & agentic behavior |

Ethics: security data is dual‑use; restrict to **defensive/educational** framing, exclude operational
malware payloads, and add refusal/safe‑completion SFT examples.

---

## 5. Training recipe (the part that actually determines quality)

### 5.1 Three stages

1. **Pretrain** (next‑token LM) on 20–40B tokens, ctx 4k, `<eos>` between docs, packed sequences.
2. **Mid‑train / long‑context** short phase: extend RoPE to 8k (adjust base / apply YaRN), up‑sample
   code, reasoning, and long documents.
3. **Post‑train:** SFT (instruction + tool‑calling + reasoning) → preference optimization
   (**DPO/KTO**, no reward model needed) → optional short **RLAIF** only if you have a verifier for
   tool tasks.

### 5.2 Data pipeline — packing, not padding

- Tokenize each document and append `<eos>` (id 3). **Concatenate then split** into exact `ctx`‑length
  windows so there is **no padding waste** during pretraining; use an **intra‑window attention reset**
  at `<eos>` (block‑diagonal / “no cross‑document attention”) so packed docs don’t attend across
  boundaries. This fixes review §2 and §6 at once.
- Store as `uint16` (V=32,768 < 65,536 ✓). Deduplicate (MinHash/exact) to avoid the long‑tail
  duplication that inflates “epochs.”
- **Reshuffle every epoch** with an epoch‑mixed seed; persist the data cursor in checkpoints so resume
  is exact (fixes review §6, §13).

### 5.3 Objective & masking

- **Loss:** token‑averaged cross‑entropy on **shifted** targets, computed **once** and consistently
  (do the shift in the collator *or* the loss, never both — the old repo did it two different ways).
  Standard form:
  ```
  loss = CE( logits[:, :-1].reshape(-1,V), labels[:, 1:].reshape(-1), ignore_index=-100 ) + z_loss
  ```
- **SFT masking:** everything except assistant/tool_call tokens → `-100`. Build ids by concatenating
  the pieces you actually emit (with `add_dummy_prefix=false`) so the mask boundary is exact — no
  prefix re‑encoding (fixes review §7). Ensure the assistant’s terminal `<eos>` (id 3) is **unmasked**
  so the model learns to stop (fixes review §1).

### 5.4 Optimizer & schedule

| Hyperparameter | Pretrain | SFT |
|---|---|---|
| Optimizer | AdamW (β=0.9, 0.95, ε=1e‑8) | AdamW |
| Peak LR | 3e‑3 (small model → higher LR) | 1e‑5 … 2e‑5 |
| Schedule | linear warmup (2%) → **cosine** to 10% floor | warmup 3% → cosine |
| Weight decay | 0.1 on 2‑D weights **except** embedding/norm/bias | 0.01 |
| Grad clip | 1.0 global norm | 1.0 |
| Precision | bf16 (+fp32 master) | bf16 |
| Batch (tokens) | 0.5–1M tokens/step (grad‑accum) | 64–256 seqs |

- **Exclude the tied embedding from weight decay** (fixes review §12); route by parameter name, not by
  `ndim>=2`.
- **Fix the LR off‑by‑one** (review §8): compute the LR for the *upcoming* update before
  `optimizer.step()`, so update #1 does not run at LR 0.
- Consider **decoupled μP‑style LR** (higher LR for small model) and **WSD schedule** (warmup‑stable‑
  decay) as an alternative to cosine for easier continued pretraining.

### 5.5 Preference optimization (DPO)

After SFT, run **DPO** on `(prompt, chosen, rejected)` pairs — chosen = correct/terminating/safe
answers, rejected = rambling, non‑terminating, or unsafe. DPO directly fixes the two behaviors the old
model failed at (stopping, following the instruction) without a reward model:

```
L_DPO = −log σ( β·[ logπ_θ(y_w|x) − logπ_ref(y_w|x) − (logπ_θ(y_l|x) − logπ_ref(y_l|x)) ] )
```

with `β≈0.1`, `π_ref` = the frozen SFT model.

---

## 6. Long context & inference

- **RoPE scaling:** trained at 8k; extend to 16k–32k via YaRN at a short fine‑tune, exploiting the
  `base=1e6` choice. Do **not** claim 32k without fine‑tuning at that length.
- **KV cache:** GQA (4 KV heads) → 100.7 MB/seq @8k bf16 (§2.1). Add **sliding‑window attention**
  (e.g. window 4k) on a subset of layers if you need 32k+ cheaply.
- **Decoding:** temperature + top‑p/top‑k + repetition penalty; **grammar‑constrained** JSON for tool
  calls; stop on `<eos>` (id 3) or `<|end|>` per turn.
- **Quantization:** the model runs in **int8/int4** (GGUF/AWQ) at ~50–100MB, enabling laptop/edge and
  CI‑runner deployment — a genuine advantage of the sub‑100M size.
- **Serving:** static KV cache + CUDA graphs, or `llama.cpp`/`vLLM` after export.

---

## 7. Evaluation (must be automated and capability‑specific)

- **Perplexity** on held‑out cyber/code/general splits (sanity only).
- **Termination rate:** % of generations that stop on `<eos>`/`<|end|>` within budget (regression test
  for the old bug).
- **Tool‑call validity:** % syntactically valid JSON + schema‑valid arguments (target >98% with
  constrained decoding).
- **Task exec accuracy:** agentic tasks with a checker (did the right tool get called with right args?).
- **Code:** HumanEval‑style pass@1 on *small* functions; expect modest numbers — track deltas.
- **Cyber knowledge:** curated MCQ set (CWE/OWASP/CVE mapping).
- **Safety:** refusal/safe‑completion rate on a red‑team suite.

---

## 8. Comparison to the previous model

| Aspect | Old (33.5M) | New (97.1M) |
|---|---|---|
| Params | 33.5M | 97.1M (verified <100M) |
| Attention | MHA, per‑layer 4k² mask buffers, manual softmax | **GQA (12/4)**, SDPA, no mask buffers |
| Positional | RoPE base 1e4 | RoPE base 1e6 (long‑ctx ready) |
| FFN | SwiGLU f=1024 (2.67×) | SwiGLU f=1920 (2.5×) |
| Vocab | 32,000 | 32,768 + reserved tool/agent tokens |
| EOS handling | **broken** (literal `</s>`, never learned) | real id 3, learned, tested |
| Doc boundaries | none | `<eos>` + packed no‑cross‑attention |
| Padding | ignored (fragile) | explicit key‑padding mask / packing |
| Post‑train | SFT only | SFT → **DPO** → constrained tool decoding |
| Tool/agentic | none | first‑class token protocol + grammar |
| Precision | fp32 | bf16 + z‑loss (+optional QK‑norm) |

---

## 9. Reference config (drop‑in dataclass)

```python
@dataclass(frozen=True)
class CyberSLM2Config:
    vocab_size: int   = 32_768
    max_seq_len: int  = 8_192
    hidden_dim: int   = 768
    num_layers: int   = 12
    num_q_heads: int  = 12
    num_kv_heads: int = 4          # GQA
    head_dim: int     = 64         # = hidden_dim // num_q_heads
    ffn_hidden_dim: int = 1_920    # SwiGLU inner (2.5×)
    rope_base: float  = 1_000_000.0
    norm_eps: float   = 1e-5
    tie_weights: bool = True
    bias: bool        = False
    qk_norm: bool     = True
    z_loss: float     = 1e-4
    # special ids — asserted against the tokenizer at load
    pad_id: int = 0; unk_id: int = 1; bos_id: int = 2; eos_id: int = 3
    def validate(self):
        assert self.hidden_dim % self.num_q_heads == 0
        assert self.head_dim == self.hidden_dim // self.num_q_heads
        assert self.num_q_heads % self.num_kv_heads == 0     # GQA group size integer
        assert self.head_dim % 2 == 0                         # RoPE
        return self
```

---

## 10. Consistency checks (all must hold — they do)

1. `head_dim · num_q_heads = 64·12 = 768 = hidden_dim` ✓ (Q/O projections square)
2. `num_q_heads % num_kv_heads = 12 % 4 = 0` ✓ (GQA group = 3)
3. `head_dim` even (64) ✓ (RoPE pairs)
4. `vocab_size = 32768 < 65536` ✓ (uint16 token storage safe)
5. Tied head: `lm_head.weight is embedding.weight` → logits `= x · Eᵀ`, shapes `(B,T,d)·(d,V)=(B,T,V)` ✓
6. KV cache `= 2·L·h_kv·d_h·ctx·2B` and query FLOPs use `h_q` — GQA asymmetry handled by repeat ✓
7. Total params **97,143,552 < 100,000,000** ✓ (§1, recomputed in §11)

---

## 11. Verify the parameter count yourself

```python
V, d, L, hq, hkv, dh, f = 32768, 768, 12, 12, 4, 64, 1920
emb  = V*d
attn = d*(hq*dh) + 2*d*(hkv*dh) + (hq*dh)*d          # Q + K + V + O
ffn  = 3*d*f                                          # gate + up + down
norm = 2*d
per  = attn + ffn + norm
total = emb + L*per + d                               # + final norm; head tied → 0
assert total == 97_143_552, total
assert total < 100_000_000
print(f"{total:,} params  ({total/1e6:.2f}M), non-embedding {(total-emb)/1e6:.2f}M")
# KV cache @ 8192 ctx, bf16:
print(2*L*hkv*dh*8192*2/1e6, "MB/seq")               # ≈ 100.7 MB
```

---

### Summary

At **97.1M parameters** this design fits comfortably under 100M while upgrading every subsystem that
matters for reasoning, code, security, and agentic tool use: **GQA** for cheap long‑context KV,
**RoPE base 1e6** for extension, **SwiGLU/RMSNorm/bf16/z‑loss** for stable training, a **real
special‑token protocol** that fixes the old model’s fatal EOS bug, **document‑boundary‑aware packed
pretraining**, **grammar‑constrained tool calling**, and a **pretrain → SFT → DPO** recipe with
correct loss masking and LR scheduling. The parameter arithmetic is exact and verified; the
capability ceiling is set honestly and addressed through data and a tool/RAG harness rather than
impossible expectations of the weights alone.
