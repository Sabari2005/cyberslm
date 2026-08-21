# Bugs found and fixed

Nineteen defects, each with the evidence that it was real and the test that
proves it is fixed. Ordered by impact.

Verified by `python cyberslm/scripts/verify.py` (35 checks) and the SFT suite
(`cyberslm_sft/tests`, 173 tests).

---

## 1. ~37% of the corpus was never trained on

`Preprocessing_Pipeline/dataloader.py`

`BinaryTokenDataset.__getitem__` drew a random start position per item from
`np.random.default_rng([seed, epoch, idx])` — sampling **with replacement**, so
one pass touched only `1 - 1/e` ≈ 63% of token positions.

Worse, the epoch was mixed in *inside* `__getitem__`, but `set_epoch()` only
mutated the main-process copy of the dataset. `make_train_dataloader` sets
`persistent_workers=nw > 0` and `num_workers=-1` resolves to `cpu_count-1`, so
worker copies were pickled once and never rebuilt — the epoch counter never
reached them and **every epoch replayed byte-identical windows**.

Measured before the fix:

```
persistent_workers = True
epoch0 starts: [444, 1516, 2942, 3315, 3369, 3385, 3641, 4469]
epoch1 starts: [444, 1516, 2942, 3315, 3369, 3385, 3641, 4469]
set_epoch changed the windows: False
```

On the real corpus that is ~72M of 194.8M tokens never seen, at any step count.

**Fix.** Windows are non-overlapping and exhaustive (`idx * context_len`), and
shuffling is delegated to the DataLoader sampler, which lives in the main
process and therefore reshuffles correctly regardless of persistent workers.
There is no epoch state left to desynchronise; `set_epoch` is now an explicit
no-op so no caller can be fooled again.

Measured after: 248/248 windows per epoch, no repeats, 100% union coverage
across epochs, and a different window dropped each epoch by `drop_last`.

## 2. Every GPU resume crashed

`cyberslm/training/checkpoint.py`

```
TypeError: RNG state must be a torch.ByteTensor
```

`load()` passes `map_location=device`. When that device is CUDA, `torch.load`
moves **every** tensor in the payload to the GPU — the RNG blob included — and
`torch.set_rng_state` accepts only a CPU `ByteTensor`. This fired on the first
real resume-after-restart. A static review cannot catch it; it needs an actual
GPU resume.

**Fix.** `_as_cpu_byte()` coerces the blob back to cpu/uint8; CUDA states are
restored only when the device count matches; the whole restore is non-fatal.
Exact data-order reproducibility is worth far less than a multi-hour run.

## 3. `best.pt` was overwritten by a worse model after every resume

`cyberslm/training/checkpoint.py`

`CheckpointManager._best_val_loss` starts at `+inf`. `Trainer._resume` restored
`self.best_val_loss` on the *Trainer*, but the manager kept its own copy, so the
first validation after a resume beat `+inf` unconditionally and clobbered the
best checkpoint. `_saved` was likewise empty, so rotation never pruned
pre-existing checkpoints.

**Fix.** `adopt_state()` re-seeds both. Verified: after resume,
`ckpt-mgr best_val == trainer best_val` and the rotation list adopts what is
already on disk.

## 4. Training in fp32 with a `dtype` setting that did nothing

`cyberslm/training/trainer.py`, `cyberslm_sft/trainer.py`

`grep autocast` matched nothing in either package. `sft_config.py` declared
`dtype: str = "float32"`, `trainer.py` logged it, and no code ever read it.

At `seq_len=4096 x batch=4 x vocab=32000` the logits tensor alone is 2.1 GB in
fp32 and `cross_entropy` materialises a second.

**Fix.** bf16 autocast with a `GradScaler` (fp16 fallback, auto-disabled on
CPU), unscaling before clipping so the grad-norm threshold means something.
Measured on A100-40GB: 182,179 tok/s, 21.73/42.4 GB.

## 5. Generation was O(n²)

`cyberslm/model/*`, both `inference.py`

The decode loop called `model.get_next_token_logits(x_cond)`, re-running all 12
layers over the entire prefix for every token. There was no `past_key_values`
plumbing at all, and `RotaryPositionEmbedding.apply` took no position offset, so
a cache could not be added without changing RoPE's signature first.

**Fix.** `offset=` on RoPE, `kv_cache`/`use_cache` through attention and block,
and `CyberSLM.generate()`. `forward()`'s `(logits, attn_weights)` contract is
untouched — the SFT adapter, the trainer and 173 tests depend on it.

Verified: cached decode matches full recomputation to **1.19e-07**, and
`generate()` matches a naive greedy loop **exactly**. 70 tok/s on CPU.

## 6. SFT trained on one tokenization and inferred with another

`cyberslm_sft/data/prompt_formatter.py` vs `cyberslm_sft/inference.py`

Training encoded each segment separately and concatenated. Inference rendered
one string via `ConversationTemplate` and encoded it once. With
`add_dummy_prefix=True` those differ at every segment boundary, so the model was
prompted off-distribution.

`conversation_template.py` claims to be "the single source of truth" and that
"the PromptFormatter delegates format-specific string construction here". It
does not — `prompt_formatter.py` imports only `configs.sft_config` and
hardcodes its own headers. Two independent implementations of one format.

**Fix.** Both inference paths go through `PromptFormatter.format_for_inference`.
Verified with the real tokenizer: the inference prompt is now an exact prefix of
the corresponding training example, and the first supervised token sits exactly
at the prompt boundary.

## 7. Inference prompts had no `### Assistant:` cue

`cyberslm_sft/data/prompt_formatter.py`

When a conversation ended on a user turn — the normal generation case — the
`else` branch did `i += 1` and emitted no assistant header, the very token
sequence the model was trained to continue from.

## 8. DDP hung at the end of every run

`cyberslm/training/trainer.py`

The final `_validate()` sat inside `if self.is_main:` but runs
`dist.all_reduce`, a collective. Non-main ranks never entered it and the run
blocked forever.

## 9. Reported SFT loss was one micro-batch out of N

`cyberslm_sft/trainer.py`

`loss_val = loss.item() * accum_steps` captured whichever micro-batch happened
to be last, then fed both `log_step` and `epoch_loss`. At `accum=4` that is 25%
of the signal reported as if it were the mean.

## 10. A skipped batch could silently drop an optimizer step

`cyberslm_sft/trainer.py`

`continue` on a fully-masked batch could land on the accumulation boundary,
skipping that optimizer step and carrying partial gradients into the next
window. Accumulation is now driven by explicit windows of micro-batches.

## 11. Short responses pulled as hard as long ones

`cyberslm_sft/trainer.py`

`masked_cross_entropy(reduction="mean")` divided by `accum_steps` weights every
micro-batch equally regardless of how many supervised tokens it holds. Now
`sum / total_tokens_in_window`, so every supervised token counts once.

## 12. Validation denominator disagreed with its numerator

`cyberslm_sft/utils/validation.py`

`n_active` counted targets in the unshifted `labels`; `masked_cross_entropy`
scores `labels[:, 1:]`. They agreed only because position 0 is always the
masked BOS.

## 13. `python train.py --help` started a real training run

`cyberslm/train.py` had no argparse at all, so the flag was ignored and `main()`
executed. Every knob is now a flag, plus `--dry-run`.

## 14. Training and inference disagreed on where checkpoints live

`train.py` wrote to `./checkpoints` relative to CWD; `inference.py` defaulted to
`cyberslm/checkpoints/best.pt`. Both now use `runs/<name>/` with an ordered
search that still finds the legacy locations.

## 15. Logging was silently discarded inside the Modal container

`infra/modal_app.py`

`Trainer` and `CheckpointManager` report through `logging.getLogger()`. The
Modal function calls `Trainer` directly rather than through `train.py:main()`,
so nothing ever called `basicConfig`, and with no root handler Python drops
everything below WARNING. The first launch trained correctly and reported
nothing — it looked frozen.

## 16. RoPE tables were built twelve times

`cyberslm/model/attention.py` documented "RoPE cache — shared across all layers"
while constructing a fresh `RotaryPositionEmbedding` per layer. 12 MB of
identical buffers, now 1.05 MB.

## 17. Pretraining CE had no `ignore_index`

Harmless on packed data, but `<pad>` is id 0 — a real vocab row — so any future
padded batch would have trained on it.

## 18. `Path.rename` is not an atomic overwrite on Windows

`checkpoint.py` used `tmp.rename(dst)`, which raises `FileExistsError` on
Windows when the destination exists. Now `os.replace`. `best.pt` also copies the
file just written instead of re-serialising ~400 MB a second time.

## 19. Dead code that tests were validating

`cyberslm/model/mask.py` (`CausalMask`, `build_causal_mask`) was exported and
exercised by `verify_phase2.py` and `verify_all.py`, but the model never called
it — `attention.py` builds its own triangular mask inline. The causal-masking
"verification" was testing a class outside the model's path. Deleted, and the
five phase scripts replaced by one suite that tests the shipped code.

---

## Not bugs

Checked and found correct, so they need no further attention:

- `_top_p_filter`'s odd-looking `scatter` — correct, because `sorted_idx` from a
  1-D sort is a full permutation, so every position is overwritten.
- Depth-scaled init reaches both residual projections (measured std 0.00689 and
  0.00713 against an expected 0.00707).
- Parameter count is exactly 33,531,264, matching the documented figure.
- Fused and explicit attention paths agree to 6e-08.
- `dataset_builder.py` splits at document boundaries, so no context window
  straddles the train/val boundary.
- Right-padded forward passes produce no NaNs on torch 2.13.
- Eight test failures that existed before any of this work (`model.embed`,
  `model.qkv`, `model.num_parameters`, literal `</s>`) were stale tests
  asserting APIs and behaviour the codebase never had. Confirmed pre-existing by
  running them against a pristine `HEAD` worktree, then updated.
