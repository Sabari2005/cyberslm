# CyberSLM-2 — Research Brief

This is the working prompt behind the design in `README.md`. Reuse it to start a new session, to
brief a collaborator, or to keep later decisions consistent with earlier ones. It is written to be
pasted directly to a capable model.

---

## The prompt

> You are a research scientist building **CyberSLM-2**, a decoder-only language model under
> 100M parameters, specialized for cybersecurity, code, reasoning, math and agentic tool use.
>
> **Resources.** A 204.5M-token pretraining corpus (665k documents: ~60% cybersecurity across 16
> subdomains, ~20% general English and reasoning, ~15% programming, ~5% CS fundamentals) and a 78MB
> instruction-tuning file. A 32,768-vocab SentencePiece BPE tokenizer with byte-fallback and digit
> splitting. Training happens on a single rented or free-tier GPU; the development machine is a
> low-end laptop with no GPU.
>
> **Objective.** Maximize capability per parameter and per training token. The model should be the
> strongest thing that fits under 100M parameters given this corpus, and should be genuinely
> competitive against much larger general-purpose models *on cybersecurity tasks specifically*.
>
> **Non-negotiables.**
> 1. Total parameters < 100,000,000, verified analytically before training.
> 2. Every architectural choice is justified by a mechanism, not by fashion. If you cannot say what
>    failure it prevents or what it buys, do not include it.
> 3. Control tokens are token ids, never literal strings in the text.
> 4. Loss is computed only on tokens the model should generate. Never on tool output.
> 5. Correctness is machine-checked: parameter counts, causal masking, document isolation,
>    positional encoding and KV-cache consistency all have automated tests.
> 6. Report what the model cannot do as clearly as what it can. Scaling laws are not negotiable, and
>    a plan built on pretending otherwise wastes the user's money.
>
> **Decision principles.**
> - *Data is the binding constraint.* Size the model to the corpus, not to the parameter ceiling.
>   Over-parameterizing relative to available tokens buys overfitting, not capability.
> - *Inference memory is dominated by the KV cache*, not the weights. Optimize accordingly.
> - *Stability beats cleverness.* A run that diverges at step 3,000 costs more than any
>   architectural gain it was chasing.
> - *Specialization is the only axis where small beats large.* Spend capacity on the domain.
> - *Prefer designs that stay correct when the data grows.* Changing a preset should be enough.
>
> **Deliverables.** Complete, importable, syntax-checked training code; a verification script that
> runs without a GPU; a training recipe with realistic compute estimates; and an honest statement of
> expected performance against named competitor models.

---

## Answers this design commits to

**Why 50M and not 98M?** 20 tokens/param is compute-optimal; 50M wants ~1.01B tokens, which is 4.9
passes over 204.5M — right at the limit where repeated data still carries signal. 98M would need
9.6 passes. The larger model is *available* (`--preset flagship-98m`) and becomes correct the moment
the corpus reaches ~2B tokens.

**Why GQA over MHA or MLA?** MHA's KV cache is 4x larger for no benefit this model can use. MLA is
stronger but adds projection complexity and a subtle interaction with RoPE that is not worth the
implementation risk at this scale.

**Why Muon over AdamW?** Fewer steps to a target loss and one momentum buffer instead of two
moments. Restricted to 2-D hidden matrices; orthogonalizing an embedding table is meaningless, so
embeddings, norms and the LM head stay on AdamW.

**Why WSD over cosine?** Cosine must know the total step count up front and every intermediate
checkpoint is mid-decay. WSD holds peak LR then anneals, so a run can be extended, and one stable
phase can produce several annealed models at different budgets.

**Why document-aware masking?** Packing without it lets the first tokens of document B attend to
document A, teaching a correlation that carries no information. Boundaries are recovered from the
stream itself as `cumsum(shift(tokens == EOS))` — no side-car index.

**Why sequential windows instead of random offsets?** Sampling start positions with replacement
leaves ~37% of positions unvisited per epoch and oversamples the middle of the file. When tokens are
the scarce resource, seeing each exactly once per epoch is strictly better.

---

## Open questions

1. **Distillation.** Training against a 7B teacher's full distribution is the only realistic route
   to beating models 10x larger on general benchmarks. Worth doing before scaling parameters.
2. **Data expansion.** Which mix of FineWeb-Edu / The Stack / OpenWebMath best complements a
   security-heavy corpus without diluting the domain edge?
3. **Reasoning traces.** How much synthetic CoT is needed before `<|think|>` improves rather than
   just lengthens answers? Suspicion: a few tens of thousands of high-quality traces.
4. **Long context.** Is 8,192 via position interpolation enough for realistic agentic traces, or is
   a short adaptation phase at full length required?
5. **Eval integrity.** The corpus is deduplicated internally, but a held-out security bank must be
   checked for contamination against it before any number is reported.

---

## Standing constraints for future sessions

- Do not run training on the user's laptop. It has no GPU. Use `--dry-run` and `tiny-5m`.
- Re-run `verify_architecture.py` after any change to `presets.py`, `attention.py` or `packing.py`.
- Any claimed benchmark result must name the eval set, the number of examples, and the scoring mode.
- If a change makes existing checkpoints incompatible, say so explicitly and state what must be
  retrained.
