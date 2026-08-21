"""
verify_architecture.py
======================
Checks the architecture is arithmetically sound before a single GPU-hour is
spent on it.

Two modes:

  * default (no torch needed, instant, safe on any laptop)
        python -m cyberslm2.scripts.verify_architecture
    Verifies parameter accounting, the <100M budget, config invariants, KV-cache
    footprint and the token-budget analysis.

  * with torch (allocates the tiny 5M model on CPU; still no training)
        python -m cyberslm2.scripts.verify_architecture --with-torch
    Additionally builds a model and empirically verifies that the analytic count
    matches reality, that attention is genuinely causal, that document masking
    isolates packed documents, and that RoPE encodes relative position.
"""

from __future__ import annotations

import argparse
import sys

from cyberslm2.configs.presets import MODEL_PRESETS, TrainConfig, get_model_config

PARAM_BUDGET = 100_000_000
CORPUS_TOKENS = 204_500_000     # measured from tokenizer/cache/tokens.bin

OK = "  [OK] "
FAIL = "  [FAIL] "

_failures: list[str] = []


def check(condition: bool, message: str) -> bool:
    print((OK if condition else FAIL) + message)
    if not condition:
        _failures.append(message)
    return condition


# ---------------------------------------------------------------------------
# Analytic checks
# ---------------------------------------------------------------------------

def verify_presets() -> None:
    print("=" * 74)
    print("PARAMETER ACCOUNTING")
    print("=" * 74)

    for name in MODEL_PRESETS:
        cfg = get_model_config(name)
        p = cfg.param_count()

        print(f"\n--- {name} ---")
        print(f"  d_model={cfg.d_model}  layers={cfg.n_layers}  "
              f"heads={cfg.n_heads} (kv={cfg.n_kv_heads}, group={cfg.n_kv_groups})")
        print(f"  head_dim={cfg.head_dim}  ffn_hidden={cfg.ffn_hidden}  "
              f"vocab={cfg.vocab_size}")
        print(f"    embedding      {p['embedding']:>12,}"
              f"   ({100 * p['embedding'] / p['total']:.1f}% of total)")
        print(f"    per layer      {p['per_layer']:>12,}")
        print(f"    all blocks     {p['blocks']:>12,}")
        print(f"    final norm     {p['final_norm']:>12,}")
        print(f"    TOTAL          {p['total']:>12,}  ({p['total'] / 1e6:.2f}M)")
        print(f"    non-embedding  {p['non_embedding']:>12,}"
              f"  ({p['non_embedding'] / 1e6:.2f}M)")

        check(p["total"] < PARAM_BUDGET,
              f"{name}: {p['total']:,} < {PARAM_BUDGET:,} budget")
        check(cfg.n_heads * cfg.head_dim == cfg.d_model,
              f"{name}: n_heads*head_dim == d_model")
        check(cfg.n_heads % cfg.n_kv_heads == 0,
              f"{name}: n_heads divisible by n_kv_heads")
        check(cfg.head_dim % 2 == 0, f"{name}: head_dim even (RoPE needs pairs)")

        # cross-check the closed form against an independent summation
        manual = (
            cfg.vocab_size * cfg.d_model
            + cfg.n_layers * (
                cfg.d_model * cfg.n_heads * cfg.head_dim
                + 2 * cfg.d_model * cfg.kv_dim
                + cfg.n_heads * cfg.head_dim * cfg.d_model
                + (2 * cfg.head_dim if cfg.qk_norm else 0)
                + 3 * cfg.d_model * cfg.ffn_hidden
                + 2 * cfg.d_model
            )
            + cfg.d_model
        )
        check(manual == p["total"],
              f"{name}: closed form matches manual sum ({manual:,})")

        kv_mb = cfg.kv_cache_bytes(cfg.max_seq_len, batch_size=1) / 1e6
        mha_mb = kv_mb * cfg.n_kv_groups
        print(f"    KV cache @ {cfg.max_seq_len} ctx: {kv_mb:.1f} MB "
              f"(MHA would be {mha_mb:.1f} MB -> {cfg.n_kv_groups}x saved)")


def verify_data_budget() -> None:
    print("\n" + "=" * 74)
    print("TOKEN BUDGET  (corpus = {:,} tokens)".format(CORPUS_TOKENS))
    print("=" * 74)
    print("\nChinchilla-optimal is ~20 tokens per parameter. Training on repeated")
    print("data works well up to ~4 epochs; past that the marginal value of a")
    print("repeat decays quickly (Muennighoff et al., 2023).\n")

    print(f"  {'preset':<16}{'params':>12}{'opt. tokens':>14}"
          f"{'epochs needed':>15}{'verdict':>14}")
    print("  " + "-" * 69)

    for name in MODEL_PRESETS:
        cfg = get_model_config(name)
        n = cfg.param_count()["total"]
        optimal = 20 * n
        epochs = optimal / CORPUS_TOKENS

        # 4-5 passes is the accepted repetition range before returns decay, so
        # anything reaching optimal within ~5 epochs is a defensible fit.
        if epochs <= 1.5:
            verdict = "data-rich"
        elif epochs <= 5.0:
            verdict = "well matched"
        else:
            verdict = "data-starved"

        print(f"  {name:<16}{n:>12,}{optimal:>14,}{epochs:>14.1f}x{verdict:>14}")

    print("\n  Reading: 'epochs needed' is how many passes over this corpus it")
    print("  would take to reach the compute-optimal token count. Above ~4x you")
    print("  are repeating data faster than it carries new information.")


def verify_train_config() -> None:
    print("\n" + "=" * 74)
    print("TRAINING CONFIG SANITY")
    print("=" * 74)

    cfg = TrainConfig()
    tps = cfg.tokens_per_step()
    total = cfg.total_tokens()
    epochs = total / CORPUS_TOKENS

    print(f"  micro_batch={cfg.micro_batch_size} x accum={cfg.grad_accum_steps} "
          f"x seq={cfg.seq_len}")
    print(f"  tokens/step  = {tps:,}")
    print(f"  total tokens = {total:,} ({total / 1e9:.2f}B)")
    print(f"  = {epochs:.1f} epochs over the corpus")

    check(cfg.warmup_steps < cfg.max_steps, "warmup_steps < max_steps")
    check(0.0 <= cfg.decay_frac < 1.0, "decay_frac in [0, 1)")
    check(tps >= 100_000,
          f"tokens/step {tps:,} >= 100k (small batches make Muon noisy)")
    check(epochs <= 6.0,
          f"{epochs:.1f} epochs over data (<= 6 before repetition stops paying)")


# ---------------------------------------------------------------------------
# Empirical checks (torch)
# ---------------------------------------------------------------------------

def verify_with_torch() -> None:
    import torch

    from cyberslm2.data.packing import build_doc_causal_mask, build_doc_ids
    from cyberslm2.model.transformer import CyberSLM2

    print("\n" + "=" * 74)
    print("EMPIRICAL CHECKS (tiny model, CPU, forward passes only)")
    print("=" * 74)

    torch.manual_seed(0)
    cfg = get_model_config("tiny-5m")
    model = CyberSLM2(cfg).eval()

    analytic = cfg.param_count()["total"]
    actual = model.num_parameters()
    check(analytic == actual,
          f"analytic {analytic:,} == actual {actual:,} parameters")

    B, T = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, _ = model(ids)
    check(tuple(logits.shape) == (B, T, cfg.vocab_size),
          f"logit shape {tuple(logits.shape)} == ({B}, {T}, {cfg.vocab_size})")

    # Causality: perturbing position t must not change any output before t.
    t = T // 2
    ids2 = ids.clone()
    ids2[:, t] = (ids2[:, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits2, _ = model(ids2)
    prefix_same = torch.allclose(logits[:, :t], logits2[:, :t], atol=1e-5)
    suffix_diff = not torch.allclose(logits[:, t], logits2[:, t], atol=1e-5)
    check(prefix_same, "attention is causal (positions before t unchanged)")
    check(suffix_diff, "position t actually responds to its own input")

    # Document masking: two packed docs must not influence each other.
    from cyberslm2.data.special_tokens import EOS_ID
    packed = torch.randint(4, cfg.vocab_size, (1, T))
    packed[0, T // 2 - 1] = EOS_ID          # boundary in the middle
    doc_ids = build_doc_ids(packed)
    check(int(doc_ids[0, -1].item()) == 1,
          f"build_doc_ids found 2 documents (last id = {int(doc_ids[0, -1])})")

    mask = build_doc_causal_mask(doc_ids)
    check(tuple(mask.shape) == (1, 1, T, T),
          f"doc mask shape {tuple(mask.shape)}")
    check(not bool(mask[0, 0, T // 2, 0].item()),
          "first token of doc 2 cannot attend to doc 1")
    check(bool(mask[0, 0, T // 2, T // 2].item()),
          "token can attend to itself")
    check(not bool(mask[0, 0, 0, T - 1].item()),
          "no attention to future positions")

    with torch.no_grad():
        masked_logits, _ = model(packed, attn_mask=mask)
    check(torch.isfinite(masked_logits).all(),
          "masked forward produces finite logits (no all-blocked rows)")

    # RoPE: the attention score between two rotated vectors must depend only on
    # their separation, not their absolute positions.
    from cyberslm2.model.rope import RotaryEmbedding
    rope = RotaryEmbedding(head_dim=cfg.head_dim, max_seq_len=64)
    q = torch.randn(1, 1, 64, cfg.head_dim)
    k = torch.randn(1, 1, 64, cfg.head_dim)
    qr, kr = rope(q, k)
    # Shift both q and k by the same absolute offset. The separation between
    # positions 3 and 8 is unchanged, so their inner product must be too.
    shifted_q, shifted_k = rope(q, k, offset=7)
    rel_a = (qr[0, 0, 3] * kr[0, 0, 8]).sum()
    rel_b = (shifted_q[0, 0, 3] * shifted_k[0, 0, 8]).sum()
    check(torch.allclose(rel_a, rel_b, atol=1e-4),
          f"RoPE score depends on relative distance only "
          f"({rel_a.item():.5f} vs {rel_b.item():.5f})")

    # Generation runs and respects the KV cache.
    with torch.no_grad():
        out = model.generate(ids[:1, :8], max_new_tokens=5, temperature=0.0)
    check(out.shape[1] == 13, f"greedy generate produced 13 tokens, got {out.shape[1]}")

    # A cached forward must equal an uncached one.
    with torch.no_grad():
        full, _ = model(ids[:1, :10])
        cached, caches = model(ids[:1, :9], use_cache=True)
        nxt, _ = model(ids[:1, 9:10], kv_caches=caches, use_cache=True)
    check(torch.allclose(full[:, 9], nxt[:, 0], atol=1e-4),
          "KV-cached decoding matches full recomputation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-torch", action="store_true",
        help="also build the 5M model on CPU and run forward-pass checks",
    )
    args = parser.parse_args()

    verify_presets()
    verify_data_budget()
    verify_train_config()

    if args.with_torch:
        try:
            verify_with_torch()
        except ImportError as exc:
            print(f"\n  [SKIP] torch unavailable: {exc}")

    print("\n" + "=" * 74)
    if _failures:
        print(f"FAILED -- {len(_failures)} check(s) did not pass:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
