"""
CyberSLM Evaluation
===================
Measures a checkpoint on held-out data and, optionally, against a baseline
checkpoint on exactly the same batches.

    python cyberslm/scripts/evaluate.py --checkpoint runs/base/best.pt
    python cyberslm/scripts/evaluate.py --checkpoint runs/base/best.pt \
        --baseline cyberslm/checkpoints/best.pt

Runs on CPU. Everything reported here is measured, not estimated:

  val loss / perplexity   token-weighted over N identical batches
  bits per token          val_loss / ln(2)
  top-1 / top-5 accuracy  next-token prediction on the same batches
  generation samples      greedy + sampled, with timing

When --baseline is given, both models are scored on the SAME batches (same
seed, same windows) so the comparison isolates the model rather than the data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "Preprocessing_Pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cyberslm.model.config import CyberSLMConfig, default_config  # noqa: E402
from cyberslm.model.model import build_model, count_parameters    # noqa: E402

PROMPTS = [
    "SQL injection is",
    "A buffer overflow occurs when",
    "The purpose of a firewall is to",
    "To detect a port scan, an analyst should",
    "AES is a symmetric cipher that",
    "Cross-site scripting allows an attacker to",
]


def load(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "config" in payload:
        cfg = CyberSLMConfig(**payload["config"])
        state = payload["model_state"]
    elif isinstance(payload, dict) and "model_state" in payload:
        cfg, state = default_config(), payload["model_state"]
    else:
        cfg, state = default_config(), payload
    model = build_model(cfg, device=device)
    model.load_state_dict(state)
    model.eval()
    meta = {
        "step": payload.get("step") if isinstance(payload, dict) else None,
        "val_loss_at_save": payload.get("val_loss") if isinstance(payload, dict) else None,
        "params": count_parameters(model)["total"],
        "seq_len": cfg.max_seq_len,
        "vocab": cfg.vocab_size,
    }
    return model, cfg, meta


@torch.no_grad()
def score(model, batches, device) -> dict:
    """Token-weighted loss plus top-1/top-5 accuracy over fixed batches."""
    nll = 0.0
    n = 0
    top1 = 0
    top5 = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        flat = logits.reshape(-1, logits.size(-1)).float()
        tgt = y.reshape(-1)
        nll += F.cross_entropy(flat, tgt, reduction="sum").item()
        n += tgt.numel()
        top = flat.topk(5, dim=-1).indices
        top1 += (top[:, 0] == tgt).sum().item()
        top5 += (top == tgt.unsqueeze(1)).any(dim=1).sum().item()
    loss = nll / n
    return {
        "val_loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_token": loss / math.log(2),
        "top1_acc": top1 / n,
        "top5_acc": top5 / n,
        "tokens_scored": n,
    }


@torch.no_grad()
def samples(model, sp, device, max_new=60) -> list[dict]:
    out = []
    for prompt in PROMPTS:
        ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        g = model.generate(x, max_new_tokens=max_new, temperature=0.0,
                           eos_id=sp.eos_id())
        dt = time.perf_counter() - t0
        new = g[0, len(ids):].tolist()
        if sp.eos_id() in new:
            new = new[: new.index(sp.eos_id())]
        out.append({
            "prompt": prompt,
            "greedy": sp.decode(new),
            "tokens": len(new),
            "tok_per_sec": len(new) / dt if dt > 0 else 0.0,
        })
    return out


def repetition_rate(text: str, n: int = 8) -> float:
    """Fraction of n-grams that are duplicates. High = degenerate looping."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--baseline", default=None,
                    help="second checkpoint to score on the SAME batches")
    ap.add_argument("--val-bin", default=str(_REPO_ROOT / "tokenizer" / "data" / "val.bin"))
    ap.add_argument("--tokenizer",
                    default=str(_REPO_ROOT / "tokenizer" / "tokenizer_output" / "tokenizer.model"))
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--context", type=int, default=None,
                    help="Score at this context instead of each model's max_seq_len. "
                         "Use it when comparing models trained at different context "
                         "lengths -- longer context lowers loss on its own, so scoring "
                         "each at its own max would flatter the wider one.")
    ap.add_argument("--spread", action="store_true",
                    help="Sample windows evenly across the WHOLE validation file "
                         "instead of taking the first N. Without this the score "
                         "reflects only the head of val.bin, which is not "
                         "representative of the split as a whole.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    import sentencepiece as spm
    from dataloader import SequentialTokenDataset

    model, cfg, meta = load(Path(args.checkpoint), device)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    print("=" * 68)
    print("  CyberSLM evaluation")
    print("=" * 68)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  params     : {meta['params']:,}")
    print(f"  trained to : step {meta['step']}  (val_loss at save: {meta['val_loss_at_save']})")
    print(f"  context    : {meta['seq_len']}   vocab: {meta['vocab']:,}")
    print(f"  device     : {device}")

    # Fixed batches, built once, reused for every model scored.
    ctx = args.context or cfg.max_seq_len
    if ctx > cfg.max_seq_len:
        raise SystemExit(f"--context {ctx} exceeds model max_seq_len {cfg.max_seq_len}")
    ds = SequentialTokenDataset(Path(args.val_bin), context_len=ctx)
    want = min(args.batches * args.batch_size, len(ds))
    if args.spread and len(ds) > want:
        # Evenly spaced indices across the entire split. Deterministic, so the
        # baseline is scored on exactly the same windows.
        step_i = len(ds) / want
        idxs = sorted({int(i * step_i) for i in range(want)})
    else:
        idxs = list(range(want))
    flat = [ds[i] for i in idxs]
    n = len(flat)
    batches = [
        (torch.stack([flat[i + j][0] for j in range(args.batch_size)]),
         torch.stack([flat[i + j][1] for j in range(args.batch_size)]))
        for i in range(0, n - args.batch_size + 1, args.batch_size)
    ]
    print(f"  scoring    : {len(batches)} batches x {args.batch_size} x {ctx} tokens"
          + ("  (context overridden)" if args.context else ""))
    print(f"  windows    : {n} of {len(ds)} "
          + ("spread across the whole split" if args.spread else "from the head of the split"))

    print("\n" + "-" * 68 + "\n  HELD-OUT METRICS\n" + "-" * 68)
    t0 = time.perf_counter()
    main_scores = score(model, batches, device)
    print(f"  val loss        : {main_scores['val_loss']:.4f}")
    print(f"  perplexity      : {main_scores['perplexity']:.2f}")
    print(f"  bits/token      : {main_scores['bits_per_token']:.4f}")
    print(f"  top-1 accuracy  : {main_scores['top1_acc']:.2%}")
    print(f"  top-5 accuracy  : {main_scores['top5_acc']:.2%}")
    print(f"  tokens scored   : {main_scores['tokens_scored']:,}  "
          f"({time.perf_counter() - t0:.1f}s)")

    base_scores = None
    if args.baseline:
        print("\n" + "-" * 68 + "\n  BASELINE (same batches)\n" + "-" * 68)
        bmodel, bcfg, bmeta = load(Path(args.baseline), device)
        if bcfg.max_seq_len < ctx:
            raise SystemExit(
                f"Baseline max_seq_len {bcfg.max_seq_len} < scoring context {ctx}. "
                f"Re-run with --context {bcfg.max_seq_len} so both are scored equally."
            )
        # Identical batches for both models. A wider baseline is simply run at
        # the narrower context, which is valid (RoPE covers it) and keeps the
        # comparison about the model rather than the conditioning window.
        bbatches = batches
        base_scores = score(bmodel, bbatches, device)
        print(f"  checkpoint      : {args.baseline}  (step {bmeta['step']})")
        print(f"  val loss        : {base_scores['val_loss']:.4f}")
        print(f"  perplexity      : {base_scores['perplexity']:.2f}")
        print(f"  top-1 accuracy  : {base_scores['top1_acc']:.2%}")
        d = base_scores["val_loss"] - main_scores["val_loss"]
        pct = 100.0 * (base_scores["perplexity"] - main_scores["perplexity"]) / base_scores["perplexity"]
        print(f"\n  DELTA           : val_loss {d:+.4f}  "
              f"({'better' if d > 0 else 'worse'}), perplexity {pct:+.1f}%")
        del bmodel

    print("\n" + "-" * 68 + "\n  GENERATION (greedy)\n" + "-" * 68)
    gen = samples(model, sp, device, args.max_new_tokens)
    for g in gen:
        rep = repetition_rate(g["greedy"])
        print(f"\n  > {g['prompt']}")
        print(f"    {g['greedy'][:300]}")
        print(f"    [{g['tokens']} tok, {g['tok_per_sec']:.1f} tok/s, "
              f"8-gram repetition {rep:.1%}]")

    avg_rep = sum(repetition_rate(g["greedy"]) for g in gen) / max(len(gen), 1)
    print(f"\n  mean 8-gram repetition across prompts: {avg_rep:.1%}")
    ds.close()

    if args.json_out:
        blob = {
            "checkpoint": args.checkpoint, "meta": meta,
            "metrics": main_scores, "baseline": base_scores,
            "baseline_checkpoint": args.baseline,
            "generations": gen, "mean_repetition": avg_rep,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")

    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
