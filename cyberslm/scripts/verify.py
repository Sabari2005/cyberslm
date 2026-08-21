"""
CyberSLM Verification Suite
===========================
Machine-checks the CURRENT model + data path. Runs on CPU in seconds; no GPU
and no training required.

    python cyberslm/scripts/verify.py

Replaces verify_all.py + verify_phase{1..4}.py, which were organised around a
development-phase decomposition rather than the shipped code, and still
exercised CausalMask -- a module the model never actually called.
"""

from __future__ import annotations

import math
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "Preprocessing_Pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cyberslm.model.config import CyberSLMConfig, default_config    # noqa: E402
from cyberslm.model.model import CyberSLM, count_parameters         # noqa: E402
from cyberslm.model.rope import RotaryPositionEmbedding             # noqa: E402
from cyberslm.training.trainer import compute_lm_loss, resolve_amp  # noqa: E402

_PASS: list[str] = []
_FAIL: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    (_PASS if ok else _FAIL).append(label)
    tag = "PASS" if ok else "FAIL"
    suffix = ("  -- " + detail) if detail else ""
    print("  [" + tag + "] " + label + suffix)


def section(title: str) -> None:
    print("\n" + "=" * 66 + "\n" + title + "\n" + "=" * 66)


def tiny() -> CyberSLMConfig:
    return CyberSLMConfig(vocab_size=256, max_seq_len=64, hidden_dim=32,
                          num_layers=4, num_heads=4, head_dim=8, ffn_hidden_dim=64)


def verify_config() -> None:
    section("1. Configuration")
    cfg = default_config()
    check("default config validates", cfg.validate() is cfg)
    check("head_dim == hidden/heads", cfg.head_dim == cfg.hidden_dim // cfg.num_heads)
    check("head_dim even (RoPE pairs)", cfg.head_dim % 2 == 0)
    bad_cases = [
        (dict(hidden_dim=384, num_heads=7), "indivisible head count"),
        (dict(head_dim=63), "odd head_dim"),
        (dict(dropout=1.5), "out-of-range dropout"),
    ]
    for kwargs, why in bad_cases:
        try:
            CyberSLMConfig(**kwargs).validate()
            ok = False
        except ValueError:
            ok = True
        check("rejects " + why, ok)


def verify_params() -> None:
    section("2. Parameter accounting")
    cfg = default_config()
    model = CyberSLM(cfg)
    info = count_parameters(model)
    emb = cfg.vocab_size * cfg.hidden_dim
    per_layer = (4 * cfg.hidden_dim ** 2
                 + 3 * cfg.hidden_dim * cfg.ffn_hidden_dim
                 + 2 * cfg.hidden_dim)
    expect = emb + cfg.num_layers * per_layer + cfg.hidden_dim
    check("analytic == actual (" + format(expect, ",") + ")",
          info["total"] == expect, "actual " + format(info["total"], ","))
    check("tied head not double counted",
          model.lm_head.weight is model.embedding.weight)
    check("all params trainable", info["total"] == info["trainable"])


def verify_rope() -> None:
    section("3. RoPE")
    rope = RotaryPositionEmbedding(head_dim=8, max_seq_len=32, base=10000)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    a = (rope.apply(q, offset=0)[0, 0, 1] * rope.apply(k, offset=0)[0, 0, 3]).sum()
    b = (rope.apply(q, offset=10)[0, 0, 1] * rope.apply(k, offset=10)[0, 0, 3]).sum()
    check("inner product depends only on relative distance",
          bool(torch.allclose(a, b, atol=1e-5)),
          format(a.item(), ".6f") + " vs " + format(b.item(), ".6f"))
    check("offset actually shifts the rotation",
          not bool(torch.allclose(rope.apply(q, offset=0), rope.apply(q, offset=1))))
    try:
        rope.apply(torch.randn(1, 1, 4, 8), offset=30)
        ok = False
    except ValueError:
        ok = True
    check("rejects span past cache end", ok)
    model = CyberSLM(tiny())
    n_rope = len({id(layer.attn.rope) for layer in model.layers})
    check("one RoPE table shared by all layers", n_rope == 1, str(n_rope) + " instance(s)")


def verify_attention_and_cache() -> None:
    section("4. Causality and KV cache")
    torch.manual_seed(0)
    model = CyberSLM(tiny()).eval()
    ids = torch.randint(0, 256, (2, 12))

    base, _ = model(ids)
    perturbed = ids.clone()
    perturbed[:, -1] = (perturbed[:, -1] + 1) % 256
    other, _ = model(perturbed)
    check("changing a future token leaves earlier logits untouched",
          bool(torch.allclose(base[:, :-1], other[:, :-1], atol=1e-6)))

    caches = [None] * len(model.layers)
    hidden = model.embedding(ids)
    fresh = []
    for block, cache in zip(model.layers, caches):
        hidden, _, present = block(hidden, kv_cache=cache, use_cache=True)
        fresh.append(present)
    caches = fresh

    seq = ids
    worst = 0.0
    for _ in range(5):
        nxt = torch.randint(0, 256, (2, 1))
        seq = torch.cat([seq, nxt], dim=1)
        hidden = model.embedding(nxt)
        fresh = []
        for block, cache in zip(model.layers, caches):
            hidden, _, present = block(hidden, kv_cache=cache, use_cache=True)
            fresh.append(present)
        caches = fresh
        incremental = model.lm_head(model.final_norm(hidden))[:, -1, :]
        full, _ = model(seq)
        worst = max(worst, (incremental - full[:, -1, :]).abs().max().item())
    check("cached decode == full recompute", worst < 1e-5,
          "max diff " + format(worst, ".2e"))

    generated = model.generate(ids, max_new_tokens=6, temperature=0.0)
    naive = ids.clone()
    for _ in range(6):
        logits, _ = model(naive)
        naive = torch.cat([naive, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    check("generate() matches a naive greedy loop", bool(torch.equal(generated, naive)))

    mask = torch.ones(2, 12, dtype=torch.long)
    mask[1, 8:] = 0
    padded, _ = model(ids, attention_mask=mask)
    check("right-padded forward stays finite", bool(torch.isfinite(padded).all()))


def verify_loss() -> None:
    section("5. Loss")
    vocab = 32
    logits = torch.zeros(2, 5, vocab)
    targets = torch.randint(0, vocab, (2, 5))
    flat = compute_lm_loss(logits, targets).item()
    check("uniform logits give ln(V)", abs(flat - math.log(vocab)) < 1e-4,
          format(flat, ".6f") + " vs " + format(math.log(vocab), ".6f"))
    partial = targets.clone()
    partial[0, 1:] = -100
    partial[1, :] = -100
    masked = compute_lm_loss(logits, partial)
    check("ignore_index honoured (no NaN, still ln(V))",
          bool(torch.isfinite(masked)) and abs(masked.item() - math.log(vocab)) < 1e-4)


def verify_dataloader() -> None:
    section("6. Data pipeline")
    from dataloader import BinaryTokenDataset, SequentialTokenDataset
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.bin"
        n_tokens, ctx = 1000, 16
        path.write_bytes(np.arange(n_tokens, dtype="<u2").tobytes())

        ds = BinaryTokenDataset(path, context_len=ctx)
        check("window count == (n-1)//C", len(ds) == (n_tokens - 1) // ctx, str(len(ds)))
        starts = [int(ds[i][0][0]) for i in range(len(ds))]
        check("windows are non-overlapping and exhaustive",
              starts == list(range(0, len(ds) * ctx, ctx)))
        x, y = ds[3]
        check("target is input shifted by one", bool(torch.equal(x[1:], y[:-1])))
        check("dataset survives pickling (Windows spawn)",
              pickle.loads(pickle.dumps(ds)) is not None)
        check("set_epoch is an explicit no-op", ds.set_epoch(7) is None)

        val = SequentialTokenDataset(path, context_len=ctx)
        last_x, _ = val[len(val) - 1]
        check("last validation window is in bounds", last_x.shape[0] == ctx)

        # Release the mmaps so the TemporaryDirectory can be removed on Windows.
        ds.close()
        val.close()
        check("close() releases the mmap", ds._mm is None and val._mm is None)


def verify_rng_restore() -> None:
    section("8. Checkpoint RNG restore")
    from cyberslm.training.checkpoint import CheckpointManager as CM

    saved = torch.get_rng_state()
    check("saved state is a CPU uint8 tensor",
          saved.dtype == torch.uint8 and saved.device.type == "cpu")

    # torch.load(..., map_location=cuda) moves every tensor in the payload to
    # the GPU, including this one, and torch.set_rng_state then rejects it with
    # "RNG state must be a torch.ByteTensor". That broke every GPU resume.
    moved = saved.to(torch.int64)
    try:
        torch.set_rng_state(moved)
        rejected = False
    except TypeError:
        rejected = True
    check("torch rejects a non-uint8 RNG blob (the original failure)", rejected)

    fixed = CM._as_cpu_byte(moved)
    check("_as_cpu_byte coerces it back",
          fixed.dtype == torch.uint8 and fixed.device.type == "cpu")

    try:
        CM._set_rng_state({"torch": moved})
        ok = True
    except Exception:
        ok = False
    check("_set_rng_state survives a coerced blob", ok)

    try:
        CM._set_rng_state({"torch": "garbage", "python": ("bogus",)})
        ok = True
    except Exception:
        ok = False
    check("_set_rng_state is non-fatal on garbage", ok)

    try:
        CM._set_rng_state({"cuda": [saved, saved, saved]})
        ok = True
    except Exception:
        ok = False
    check("mismatched CUDA device count does not raise", ok)


def verify_amp() -> None:
    section("7. Mixed precision")
    cpu = torch.device("cpu")
    dtype, scaler = resolve_amp("bfloat16", cpu)
    check("AMP disabled on CPU", dtype is None and scaler is False)
    dtype, _ = resolve_amp("float32", cpu)
    check("float32 means no autocast", dtype is None)
    try:
        resolve_amp("nonsense", cpu)
        ok = False
    except ValueError:
        ok = True
    check("rejects unknown dtype", ok)


def main() -> int:
    print("CyberSLM verification -- CPU only, no training performed")
    verify_config()
    verify_params()
    verify_rope()
    verify_attention_and_cache()
    verify_loss()
    verify_dataloader()
    verify_rng_restore()
    verify_amp()

    section("SUMMARY")
    print("  passed: " + str(len(_PASS)) + "    failed: " + str(len(_FAIL)))
    for name in _FAIL:
        print("    FAILED -- " + name)
    print("=" * 66)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
