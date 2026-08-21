# CyberSLM — Code Review: Mistakes & How to Fix Them

**Reviewed:** the full pipeline — tokenizer (`tokenizer/`, `Preprocessing_Pipeline/`), pretraining
(`cyberslm/`), and supervised fine-tuning (`cyberslm_sft/`) — plus the real training logs
(`cyberslm/checkpoints/report.txt`, `cyberslm_sft/checkpoints/train.log`) and dataset stats.

This document lists concrete defects, ranked by severity, each with **file:line**, **what it does**,
**why it is wrong**, and **the fix**. A companion file, `NEW_ARCHITECTURE.md`, proposes a
mathematically-verified redesign.

---

## TL;DR — the single most important bug

**The tokenizer's end-of-sequence token and the string the model is trained to emit do not match.**

- The tokenizer was built with control pieces `<bos>`=2 and `<eos>`=3
  (`tokenizer/train_tokenizer.py:125-128`, confirmed in `tokenizer/tokenizer_output/tokenizer.vocab`:
  `<pad> 0 / <unk> 1 / <bos> 2 / <eos> 3`).
- But the SFT template appends the **literal string** `"</s>"` and prepends `"<s>"`
  (`cyberslm_sft/configs/sft_config.py:201`, `data/conversation_template.py:52`). SentencePiece
  encodes `</s>` as ordinary characters (`<`, `/`, `s`, `>`), **not** as token id 3.
- Generation stops on `eos_id = 3` (`cyberslm_sft/utils/inference.py:229`), which the model was
  **never trained to emit**.

**Proof in the logs:** every SFT sample response ends with the literal text `</s>` and then keeps
going (`cyberslm_sft/checkpoints/train.log:712`, `719`, …). The model never produces token 3, so
generation runs to `max_new_tokens` every time. This alone makes the instruct model unusable for
real serving. Fix in §1.

---

## Measured facts (from the repo, for grounding)

| Fact | Value | Source |
|---|---|---|
| Pretraining corpus | 204,501,030 tokens, 665,067 docs | `tokenizer/stats.json`, `tokenize.log` |
| Model size | 33,531,264 params (~33.5M) | `cyberslm/checkpoints/report.txt:7` |
| Pretrain tokens *seen* | ~524M (4000 steps × 131,072) | `report.txt:492` |
| Final pretrain val loss | 2.427 (ppl ≈ 11.3) | `report.txt:493` |
| SFT data | 24,979 samples (all conversation format) | `train.log:146-148` |
| SFT final val loss | 2.3719 (ppl ≈ 10.72) | `train.log:701` |
| SFT active-token ratio | 51.9% | `train.log:698` |
| Docs with NO EOS separator | 665,067 (EOS never inserted) | `tokenize.log` totals match stats exactly |

The corpus is ~204M tokens. A 33.5M-parameter model is *compute-starved on capacity, not tokens*:
its qualitative output (log lines 710-778) is fluent but does not follow instructions
(“What is AES?” → an answer about HTTP; “reverse a string” → a MITM description). That is a
**capability** limit addressed by `NEW_ARCHITECTURE.md`; the items below are **bugs** that should be
fixed regardless of model size.

---

## CRITICAL

### 1. EOS/BOS special-token mismatch (the model never learns to stop)

**Files:** `cyberslm_sft/configs/sft_config.py:198-201`, `data/conversation_template.py:52`,
`data/prompt_formatter.py:113,191,212`, `cyberslm_sft/utils/inference.py:132,229`.

**What it does:** training text ends each assistant turn with the 4-character string `"</s>"` and
opens with `"<s>"`. These strings are *not* in the tokenizer’s vocab as single tokens; the real
control ids are `<bos>=2`, `<eos>=3`.

**Why it is wrong:** the model is trained to predict the literal characters `<`,`/`,`s`,`>` as an
“ending,” but the sampler stops on id 3. The two never coincide, so the model cannot terminate. It
also wastes tokens and pollutes output with literal tag text.

**Fix:** stop using literal strings; append the real control id at the token level.

```python
# prompt_formatter.py — append EOS as an ID, not a string
# In _build_alpaca_labels / _build_conversation_labels, build ids explicitly:
all_ids = tokenizer.encode(prompt_only)                 # prompt part
resp_ids = tokenizer.encode(response) + [tokenizer.eos_id]   # eos_id == 3
input_ids = all_ids + resp_ids
labels    = [IGNORE_INDEX]*len(all_ids) + list(resp_ids)     # learn to emit EOS(3)
```

Set `TemplateConfig.eos_string = ""` (do not inject text), and drop the literal `"<s>"`; prepend
`tokenizer.bos_id` (2) once at the sequence start as an id. Then generation’s `next_id == eos_id`
(3) will actually fire. Add a startup assertion: `assert tokenizer.eos_id == 3 and
tokenizer.bos_id == 2`.

> Note the naming trap: the tokenizer pieces are `<bos>`/`<eos>` (angle-bos), **not** the Llama-style
> `<s>`/`</s>`. Pick one convention and use the *ids* everywhere.

---

### 2. Pretraining never inserts document separators (no notion of boundaries)

**Files:** `Preprocessing_Pipeline/dataset_tokenizer.py:74-92,330-334` (`--add-bos`/`--add-eos`
default `False`), driven with neither flag — `tokenize.log` shows total tokens == `stats.json`
total exactly, so **no** EOS was added between the 665k documents.

**What it does:** all documents are concatenated into one flat stream with no separator. Training
windows (`Preprocessing_Pipeline/dataloader.py:122-133`) slice arbitrary 4096-token spans that
freely straddle unrelated documents.

**Why it is wrong:** the base model never sees an end-of-document signal, so it cannot learn to stop
or to treat documents as independent. This is the root cause of the rambling, non-terminating base
model and compounds bug #1.

**Fix:** tokenize with an inserted `<eos>` (id 3) after every document (and optionally `<bos>` before):

```bash
python dataset_tokenizer.py --input Final.jsonl --model tokenizer.model \
    --add-eos            # append id 3 per document
```

Rebuild `train.bin`/`val.bin`. Now windows contain real boundaries and the model learns termination
even during pretraining.

---

### 3. `masked_cross_entropy(reduction="none")` will crash (shape bug)

**File:** `cyberslm_sft/data/loss_masking.py:86-113`.

**What it does:** it shifts (`shift_logits = logits[..., :-1, :]`, `shift_labels = labels[..., 1:]`),
flattens to `B*(T-1)`, computes loss, then for `reduction="none"` does `loss.reshape(B, T)`.

**Why it is wrong:** after the shift the tensor has `B*(T-1)` elements, which cannot be reshaped to
`(B, T)` — it raises `RuntimeError`. It is only latent because training/validation call `"mean"`/`"sum"`
(lines 288, 108), but any diagnostic use of `"none"` breaks.

**Fix:** reshape to the post-shift length.

```python
if reduction == "none":
    loss = loss.reshape(B, T - 1)   # not (B, T)
```

*(The label-shift alignment itself is correct: labels are unshifted per-position with prompt masked,
and the model does not shift internally, so the single shift here is right. Good.)*

---

## HIGH

### 4. `attention_mask` is silently ignored; padding correctness rests on an unstated invariant

**Files:** `cyberslm_sft/model/cyberslm.py:32-39`, `data/collator.py:85-92`,
`utils/validation.py:167-170`.

**What it does:** the SFT model’s `forward(input_ids, attention_mask=None, **kwargs)` accepts a mask
but never uses it — it calls the base model, which applies only a **causal** mask
(`cyberslm/model/attention.py:183`).

**Why it is (mostly) OK today, but fragile:** with **right** padding + causal attention, real tokens
(positions `< L`) never attend to trailing PAD tokens, and PAD query rows are dropped by
`IGNORE_INDEX` labels — so results are correct *by accident of the padding side*. It is a **HIGH**
risk because: (a) any switch to left-padding, packing, or bidirectional use silently corrupts real
tokens; (b) the mask is threaded through the whole stack suggesting it works when it does not;
(c) batched generation with padding would be wrong.

**Fix:** either make padding correctness explicit, or make the mask real. Minimal, robust option —
combine the causal mask with a key-padding mask inside attention:

```python
# attention.py forward(x, attn_mask=None)
scores = scores + causal_bias
if attn_mask is not None:                 # attn_mask: (B, T), 1=keep 0=pad
    pad = (attn_mask == 0)[:, None, None, :]      # (B,1,1,T)
    scores = scores.masked_fill(pad, float("-inf"))
```

and pass it down from `DecoderBlock`/`CyberSLM.forward`. Add an assertion that padding is
right-side if you keep the shortcut.

---

### 5. Per-layer 4096×4096 causal-mask buffers waste ~0.8 GB and SDPA is not used

**Files:** `cyberslm/model/attention.py:126,177-202`, `mask.py:61-78`.

**What it does:** each of the 12 attention modules constructs its own
`CausalMask(max_seq_len=4096)` → a `(1,1,4096,4096)` float32 buffer = **67 MB each × 12 = ~805 MB**.
Attention is also computed manually (`matmul → softmax → matmul`), materialising a
`(B,H,T,T)` score tensor.

**Why it is wrong:** the mask is identical across layers (should be shared or generated on the fly),
and the manual path forgoes `torch.nn.functional.scaled_dot_product_attention`, which the file’s own
docstring recommends (`attention.py:44-53`). For `T=4096` the explicit `T×T` matrix is a large,
avoidable memory/latency cost.

**Fix:** delete `CausalMask` from the hot path and use fused SDPA:

```python
import torch.nn.functional as F
context = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.attn_dropout_p if self.training else 0.0)
```

This removes all mask buffers, enables FlashAttention kernels, and is numerically identical.

---

### 6. Random-start sampling never reshuffles across epochs (`set_epoch` never called)

**Files:** `cyberslm/training/trainer.py:311-378` (loop), `Preprocessing_Pipeline/dataloader.py:113-117`.
`BinaryTokenDataset` pre-generates start positions for epoch 0 and exposes `set_epoch()`, but the
`Trainer` never calls it; it just recreates the iterator on exhaustion
(`trainer.py:550-555`).

**What it does:** the same fixed set of `num_samples` random windows is reused every pass; positions
are drawn *with replacement* (`rng.integers`, `dataloader.py:109-111`), so some spans repeat and
others are never seen.

**Why it is wrong:** reduced data diversity and silent over/under-sampling; “epochs” are not real
epochs. With `num_samples = min(200000, (n_tokens-1)//4096) ≈ 47k` (`cyberslm/train.py:54`), coverage
is ~one nominal pass of 47k windows recycled ~11× over 524M seen tokens.

**Fix:** track an epoch counter and reshuffle, or (better) sample a fresh random offset per
`__getitem__` with an epoch-mixed seed:

```python
def __getitem__(self, idx):
    rng = np.random.default_rng((self.seed, self._epoch, idx))
    pos = int(rng.integers(0, self.max_start + 1))
    ...
# and in the trainer, before each pass: dataset.set_epoch(epoch)
```

---

### 7. Loss-mask boundary via prefix re-encoding is off-by-one under `add_dummy_prefix`

**Files:** `cyberslm_sft/data/prompt_formatter.py:288-306,325-332,350-362`; tokenizer trained with
`add_dummy_prefix=True` (`tokenizer/train_tokenizer.py:112`).

**What it does:** it finds where the response starts by encoding the prefix string *separately*
(`len(tokenizer.encode(prompt_only))`) and masking that many tokens of the full-sequence encoding.

**Why it is wrong:** with `add_dummy_prefix=True`, encoding a substring prepends a leading `▁` and can
merge boundary characters differently than the same span inside the full string, so `len(prefix_ids)`
does not necessarily equal the number of full-sequence tokens covering the prefix. The mask can drift
by 1–2 tokens per turn — occasionally training on a stray prompt token or skipping the first response
token. (The auditor’s 51.9% active ratio, `train.log:698`, is plausible but imprecise.)

**Fix:** build ids by concatenation and derive the boundary from the *pieces you actually
concatenated* (see the fix in §1), so no re-encoding/alignment guesswork is needed. If you must map
char→token, encode once and use SentencePiece offset APIs
(`encode(..., out_type='immutable_proto')` piece spans) instead of length of a re-encoded prefix.

---

### 8. First optimizer step runs at LR = 0 (scheduler off-by-one)

**File:** `cyberslm/training/scheduler.py:96-97,133-145`; used in `trainer.py:372-373`.

**What it does:** the scheduler sets LR to `get_lr(0) = peak·0/warmup = 0` at construction, and the
loop does `optimizer.step()` **before** `scheduler.step()`. So update #1 uses LR 0, and every update
uses `get_lr(step-1)` — the schedule lags by one step.

**Why it is wrong:** one fully wasted update and a consistent one-step LR lag. Minor for 4000 steps,
but incorrect and confusing when reasoning about warmup.

**Fix:** step the scheduler to compute the LR for the *upcoming* update, or initialise at
`get_lr(1)`. Simplest: call `lr = scheduler.step()` **before** `optimizer.step()`, or set the LR for
step *t* prior to applying step *t*. The SFT side uses `LambdaLR` correctly ordered
(`trainer.py:308-309`), so mirror that.

---

## MEDIUM

### 9. Base model pretrained without BOS, but inference prepends BOS

**Files:** `cyberslm/inference.py:224-228` prepends `bos_id()` (2); pretraining used neither
`--add-bos` nor a BOS in packing (§2).

**Why it matters:** position 0 at inference is a token the model never saw during training →
degraded first-step distribution. Either train with a leading BOS per document/sequence, or do not
prepend it at inference. Be consistent (the SFT path has the same issue via the literal `<s>`).

### 10. Validation only ever covers the first N windows

**File:** `cyberslm/training/trainer.py:470-511`. `_validate` builds a fresh
`iter(self.val_loader)` each call and reads `val_steps` (=100, `cyberslm/train.py`) sequential
batches, always from the start of `val.bin`. The tail of the validation set is never evaluated.

**Fix:** iterate the full val loader (or a fixed strided subset), and cache the iterator/positions so
reported val loss reflects the whole split.

### 11. Gradient-accumulation boundary step is mis-scaled

**File:** `cyberslm_sft/trainer.py:288,300-313`. Loss is divided by `gradient_accumulation_steps`
every micro-batch, but the last accumulation window of each epoch may contain fewer than
`accum` micro-batches (the `(batch_idx+1)==len(loader)` trigger, line 303). That final step’s gradient
is divided by 4 while only 1–3 micro-batches were summed → an under-weighted update. Also
`loss_val = loss.item() * accum` (line 313) logs only the **last** micro-batch loss, not the window
average.

**Fix:** divide by the *actual* number of micro-batches in the window, and accumulate the running
loss over the window before logging.

### 12. Weight decay is applied to the (tied) embedding matrix

**Files:** `cyberslm/training/trainer.py:149-156` and `cyberslm_sft/utils/optimizer.py:62-73` put all
`ndim >= 2` tensors in the decay group. The token embedding is 2-D and is the *same tensor* as the LM
head (weight-tied, `model.py:114-115`). Decaying it shrinks the output projection too.

**Why it matters:** many modern recipes exclude the embedding/unembedding from weight decay
(especially when tied) to avoid biasing logit scale. It is a defensible choice either way, but it is
currently implicit and undocumented.

**Fix:** if you want the common convention, match embeddings by name and route them to the `no_decay`
group; otherwise document the deliberate choice.

### 13. RNG-state resume is incomplete for the DataLoader

**File:** `cyberslm/training/checkpoint.py:255-283` saves torch/numpy/python/cuda RNG, but the
`BinaryTokenDataset` start positions are derived from `(seed, epoch)` and the trainer’s epoch is not
persisted (nor is `set_epoch` used, §6). A resumed run does not reproduce the same data order.

**Fix:** persist and restore the data epoch/position; combine with §6.

### 14. `float("-inf")` masks can produce NaNs on fully-masked rows

**File:** `cyberslm/model/mask.py:70-78`. Additive `-inf` is fine for causal masks (every row has at
least the diagonal), but once a **padding** mask is added (§4), a query row that is entirely padding
becomes all `-inf` → `softmax` returns NaN and poisons the batch.

**Fix:** prefer SDPA (§5), or use a large finite negative (e.g. `-1e9` / dtype min) and guarantee at
least one valid key per row.

---

## LOW / QUALITY

- **`_apply_repetition_penalty` mutates input logits in place** (`cyberslm/inference.py:111`) and the
  default `1.3` is applied inside the sampler for *all* strategies including where callers may not
  expect it (`_sample_next_token` default arg is a mutable `[]`, `inference.py:121` — classic mutable
  default-argument smell; harmless here only because it is reassigned per call).
- **Two different, undocumented loss conventions** coexist: pretraining relies on the *dataloader*
  pre-shifting `y` (`dataloader.py:131-132`, loss does **not** shift, `trainer.py:112-117`), while SFT
  shifts **inside** the loss (`loss_masking.py:86-88`). Both are individually correct but the split is
  a foot-gun; document it prominently.
- **Tokenizer vocab spends capacity on CJK** (`tokenizer.vocab` tail shows Chinese pieces) at
  `character_coverage=0.9995` for an English/code/cyber corpus — wasted rows in a 32k vocab. Lower
  coverage or curate the corpus.
- **One 10.7M-token “document”** dominates the corpus (`stats.json:max_tokens`), and `max_sentence_length`
  in tokenizer training is 16 KB bytes — inconsistent handling of pathological docs; pre-split长 docs.
- **Determinism is partial**: seeds are set but `torch.use_deterministic_algorithms`,
  cuDNN flags, and fused-AdamW nondeterminism are not addressed.
- **`fused=torch.cuda.is_available()`** in `build_optimizer` (`trainer.py:174`) is fine on GPU but
  couples optimizer choice to device availability without a flag.
- **`count_response_tokens` counts pre-shift tokens** (`loss_masking.py:120-145`); the loss trains on
  `T-1` shifted positions, so the reported “active token” count is off by the per-sequence boundary.
  Cosmetic, but the throughput/active numbers are slightly overstated.
- **Docstring/implementation drift**: `ffn.py:13` even flags its own “WRONG shorthand,” and
  `attention.py` promises SDPA compatibility it never uses. Clean these up.

---

## What the code gets *right* (keep it)

- RoPE math is correct: interleaved even/odd rotation with `repeat_interleave(2)` cos/sin reproduces
  `x'₂ᵢ = x₂ᵢcosθ − x₂ᵢ₊₁sinθ`, `x'₂ᵢ₊₁ = x₂ᵢ₊₁cosθ + x₂ᵢsinθ` exactly (`rope.py:156-240`), computed
  in float64→float32 for stability.
- RMSNorm is standard and computed in float32 (`norm.py:79-111`).
- SwiGLU is the correct 3-matrix `SiLU(xWg) ⊙ (xWv) Wo` form (`ffn.py:97-121`).
- Pre-norm residual blocks with scaled output-projection init `1/√(2L)` (`model.py:139-150`).
- Weight tying is implemented and verified after init (`model.py:114-115,158-161`).
- Parameter counting de-duplicates the tied tensor via `data_ptr()` (`model.py:267-279`).
- SFT loss masking (label alignment) is correct; the base checkpoint loads into the SFT model with
  `strict=True` because the SFT model subclasses the pretraining model verbatim
  (`cyberslm_sft/model/cyberslm.py`).
- Atomic checkpoint writes (tmp-then-rename), best/latest tracking, and cosine+warmup schedules are
  sound.

---

## Priority fix order

1. **§1 EOS/BOS ids** and **§2 document EOS** — without these the instruct model cannot stop or
   follow the intended format. (Re-tokenize, re-pretrain or at minimum re-SFT.)
2. **§4 padding mask** made explicit + **§5 SDPA** — correctness hardening and a large
   memory/speed win.
3. **§6 data reshuffle**, **§7 mask boundary**, **§8 LR off-by-one** — training-quality correctness.
4. Everything under MEDIUM/LOW as cleanup.

After §1–§2, re-run SFT and confirm generations terminate on id 3 and no literal `</s>` appears in
output.
