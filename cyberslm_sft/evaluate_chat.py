"""
Evaluate the instruction-tuned CyberSLM on chat prompts.

    python cyberslm_sft/evaluate_chat.py --checkpoint runs/sft/best/model.pt

Prompts are built with the SAME PromptFormatter used in training, so the model
sees exactly the token sequence it was fine-tuned on. Reports, per prompt:

  response            greedy decode (deterministic)
  8-gram repetition   fraction of duplicated 8-grams; high means degenerate
  tokens / speed

The prompt set deliberately mixes in-domain security questions with general
knowledge and code, because a 33.5M model can plausibly learn the former while
failing the latter, and a report that only showed the former would mislead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_SFT_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SFT_ROOT.parent
for _p in (_REPO_ROOT, _SFT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from configs.sft_config import default_config          # noqa: E402
from data.prompt_formatter import PromptFormatter, Tokenizer  # noqa: E402
from model.cyberslm import CyberSLM as SFTModel        # noqa: E402

PROMPTS = [
    ("security", "What is SQL injection and how do I prevent it?"),
    ("security", "Explain the difference between symmetric and asymmetric encryption."),
    ("security", "What should I check first when investigating a possible phishing email?"),
    ("security", "What is a buffer overflow?"),
    ("general", "What is Linux?"),
    ("general", "Explain the CIA triad."),
    ("code", "Write a Python function to reverse a string."),
    ("code", "Write a Python function that checks if a port is open."),
]


def repetition_rate(text: str, n: int = 8) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(_REPO_ROOT / "runs" / "sft" / "best" / "model.pt"))
    ap.add_argument("--tokenizer",
                    default=str(_REPO_ROOT / "tokenizer" / "tokenizer_output" / "tokenizer.model"))
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy, deterministic and reproducible")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = default_config()
    cfg.tokenizer.model_path = args.tokenizer
    cfg.data.max_seq_len = 2048
    cfg.model.max_seq_len = 2048

    tok = Tokenizer(cfg.tokenizer.model_path)
    fmt = PromptFormatter(cfg=cfg, tokenizer=tok)

    model = SFTModel(cfg.model)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print("  CyberSLM instruct - chat evaluation")
    print("=" * 70)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  params     : {n_params:,}")
    print(f"  decoding   : {'greedy' if args.temperature == 0 else f'T={args.temperature}'}")
    print(f"  device     : {device}")

    results = []
    for kind, question in PROMPTS:
        ids = fmt.format_for_inference({"messages": [{"role": "user", "content": question}]})
        x = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        out = model.generate(
            x, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, eos_id=tok.eos_id,
        )
        dt = time.perf_counter() - t0
        new = out[0, len(ids):].tolist()
        if tok.eos_id in new:
            new = new[: new.index(tok.eos_id)]
        text = tok.decode(new).strip()
        rep = repetition_rate(text)
        results.append({
            "kind": kind, "prompt": question, "response": text,
            "tokens": len(new), "tok_per_sec": len(new) / dt if dt else 0.0,
            "repetition": rep, "stopped_on_eos": tok.eos_id in out[0, len(ids):].tolist(),
        })
        print("\n" + "-" * 70)
        print(f"  [{kind}] {question}")
        print("-" * 70)
        print(text[:900] if text else "(empty)")
        print(f"  [{len(new)} tok, {len(new) / dt if dt else 0:.1f} tok/s, "
              f"8-gram repetition {rep:.1%}, "
              f"{'stopped on EOS' if results[-1]['stopped_on_eos'] else 'hit token limit'}]")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for kind in ("security", "general", "code"):
        sub = [r for r in results if r["kind"] == kind]
        if not sub:
            continue
        mr = sum(r["repetition"] for r in sub) / len(sub)
        eos = sum(r["stopped_on_eos"] for r in sub)
        print(f"  {kind:<9} n={len(sub)}  mean 8-gram repetition {mr:6.1%}  "
              f"stopped on EOS {eos}/{len(sub)}")
    overall = sum(r["repetition"] for r in results) / len(results)
    print(f"  {'overall':<9} n={len(results)}  mean 8-gram repetition {overall:6.1%}  "
          f"stopped on EOS {sum(r['stopped_on_eos'] for r in results)}/{len(results)}")
    print("=" * 70)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({
            "checkpoint": args.checkpoint, "params": n_params,
            "temperature": args.temperature, "results": results,
            "mean_repetition": overall,
        }, indent=2), encoding="utf-8")
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
