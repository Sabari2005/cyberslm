"""
Turn evaluation JSON into a readable Markdown report.

    python cyberslm/scripts/make_report.py \
        --new runs/reports/base_new.json \
        --old runs/reports/baseline_old_run.json \
        --out runs/reports/REPORT.md

Only reports numbers that are present in the JSON. Anything not measured is
printed as "not measured" rather than estimated or inferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pct_delta(new: float, old: float) -> str:
    if old in (0, None) or new is None:
        return "n/a"
    return f"{100.0 * (old - new) / old:+.1f}%"


def row(label: str, new, old, fmt="{:.4f}", lower_better=True) -> str:
    if new is None:
        return f"| {label} | not measured | - | - |"
    n = fmt.format(new)
    if old is None:
        return f"| {label} | {n} | not measured | - |"
    o = fmt.format(old)
    better = (new < old) if lower_better else (new > old)
    mark = "better" if better else ("same" if new == old else "worse")
    return f"| {label} | **{n}** | {o} | {mark} |"


def parse_curve(log_path: Path) -> tuple[list[tuple[int, float]], dict]:
    """
    Pull the validation curve and throughput out of a training log.

    Returns ([(step, val_loss)], {"tok_per_sec": ..., "final_step": ...}).
    Returns empty results rather than raising if the log is absent or has a
    different shape -- a missing curve must not block the rest of the report.
    """
    import re

    if not log_path or not Path(log_path).exists():
        return [], {}
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    curve = [(int(a), float(b)) for a, b in
             re.findall(r"Validation\s+step=(\d+)\s+val_loss=([0-9.]+)", text)]
    rates = [int(r.replace(",", "")) for r in
             re.findall(r"tok/s=([\d,]+)", text)]
    steps = [int(x) for x in re.findall(r"step=\s*(\d+)\s+loss=", text)]
    meta = {}
    if rates:
        mid = sorted(rates)[len(rates) // 2]
        meta["tok_per_sec_median"] = mid
    if steps:
        meta["final_step"] = max(steps)
    return curve, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="CyberSLM base model - training report")
    ap.add_argument("--log", default=None,
                    help="training log to extract the validation curve from")
    args = ap.parse_args()

    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    old = json.loads(Path(args.old).read_text(encoding="utf-8")) if args.old else None

    nm = new["metrics"]
    nmeta = new["meta"]

    # Prefer the baseline embedded in the evaluation run: evaluate.py scored it
    # on the IDENTICAL batches. A separately-produced --old file may have used a
    # different sample of the split, and mixing the two silently compares a
    # spread sample against a head-of-file sample - which turned a real 23%
    # perplexity reduction into a fictitious 70% one.
    same_batch = new.get("baseline") is not None
    if same_batch:
        om = new["baseline"]
        ometa = (old["meta"] if old else {})
        baseline_label = new.get("baseline_checkpoint") or "previous"
    else:
        om = (old["metrics"] if old else {})
        ometa = (old["meta"] if old else {})
        baseline_label = (old["checkpoint"] if old else "n/a")
    old = old or ({"metrics": om, "meta": ometa} if om else None)

    L: list[str] = []
    L.append(f"# {args.title}\n")
    L.append("All figures below are measured on held-out data, not estimated.\n")

    L.append("## Setup\n")
    L.append("| | new | previous |")
    L.append("|---|---|---|")
    L.append(f"| checkpoint | `{new['checkpoint']}` | `{baseline_label}` |")
    L.append(f"| parameters | {nmeta.get('params', 0):,} | "
             f"{ometa.get('params', 0):,} |" if old else
             f"| parameters | {nmeta.get('params', 0):,} | n/a |")
    L.append(f"| trained to step | {nmeta.get('step')} | {ometa.get('step', 'n/a')} |")
    L.append(f"| context | {nmeta.get('seq_len')} | {ometa.get('seq_len', 'n/a')} |")
    L.append(f"| vocab | {nmeta.get('vocab', 0):,} | {ometa.get('vocab', 0):,} |"
             if old else f"| vocab | {nmeta.get('vocab', 0):,} | n/a |")
    L.append(f"| tokens scored | {nm.get('tokens_scored', 0):,} | "
             f"{om.get('tokens_scored', 0):,} |" if old else
             f"| tokens scored | {nm.get('tokens_scored', 0):,} | n/a |")
    L.append("")

    L.append("## Held-out metrics\n")
    if same_batch:
        L.append("Both models scored on the **same batches at the same context**, "
                 "so the comparison reflects the model rather than the sample or "
                 "the conditioning window." + chr(10))
    else:
        L.append("> **Caution:** the baseline figures come from a separate "
                 "evaluation run and may not use the same windows. Treat the "
                 "deltas as indicative only." + chr(10))
    L.append("| metric | new | previous | |")
    L.append("|---|---|---|---|")
    L.append(row("validation loss", nm.get("val_loss"), om.get("val_loss")))
    L.append(row("perplexity", nm.get("perplexity"), om.get("perplexity"), "{:.2f}"))
    L.append(row("bits / token", nm.get("bits_per_token"), om.get("bits_per_token")))
    L.append(row("top-1 accuracy", nm.get("top1_acc"), om.get("top1_acc"),
                 "{:.2%}", lower_better=False))
    L.append(row("top-5 accuracy", nm.get("top5_acc"), om.get("top5_acc"),
                 "{:.2%}", lower_better=False))
    L.append(row("8-gram repetition", new.get("mean_repetition"),
                 old.get("mean_repetition") if old else None, "{:.1%}"))
    L.append("")

    if nm.get("perplexity") and om.get("perplexity"):
        red = 100.0 * (om["perplexity"] - nm["perplexity"]) / om["perplexity"]
        word = "reduction" if red > 0 else "increase"
        L.append(
            f"Perplexity {word}: **{abs(red):.1f}%** "
            f"({om['perplexity']:.2f} -> {nm['perplexity']:.2f}) over "
            f"{nm.get('tokens_scored', 0):,} held-out tokens." + chr(10)
        )

    curve, cmeta = parse_curve(Path(args.log)) if args.log else ([], {})
    if curve:
        L.append("## Training curve" + chr(10))
        if cmeta.get("tok_per_sec_median"):
            L.append(f"Median throughput: **{cmeta['tok_per_sec_median']:,} tok/s**." + chr(10))
        L.append("| step | tokens seen | val loss |")
        L.append("|---:|---:|---:|")
        tps = nmeta.get("tokens_per_step", 131072)
        for st, vl in curve:
            L.append(f"| {st:,} | {st * tps / 1e6:,.0f}M | {vl:.4f} |")
        L.append("")
        if old and om.get("val_loss"):
            beat = next((st for st, vl in curve if vl < om["val_loss"]), None)
            if beat:
                L.append(
                    f"Crossed the previous model's held-out loss "
                    f"({om['val_loss']:.4f}) at **step {beat:,}** "
                    f"({beat * tps / 1e6:,.0f}M tokens)." + chr(10)
                )

    L.append("## Generation samples\n")
    L.append("Greedy decoding (temperature 0), so these are deterministic and "
             "reproducible.\n")
    for g in new.get("generations", []):
        L.append(f"**Prompt:** `{g['prompt']}`\n")
        L.append("```")
        L.append(g["greedy"][:600])
        L.append("```")
        L.append(f"*{g['tokens']} tokens at {g['tok_per_sec']:.1f} tok/s*\n")

    if old and old.get("generations"):
        L.append("### Previous model, same prompts\n")
        for g in old["generations"][:3]:
            L.append(f"**Prompt:** `{g['prompt']}`\n")
            L.append("```")
            L.append(g["greedy"][:400])
            L.append("```")
        L.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}  ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
