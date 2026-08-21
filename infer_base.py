"""
Base model inference — raw text continuation.

    python Final/infer_base.py --prompt "SQL injection is"
    python Final/infer_base.py --interactive

The base model is a *continuation* model, not a chat model. Give it the start of
a sentence and it continues. Asking it a question will not get an answer; it
will continue the question. Use infer_chat.py for question answering.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cyberslm.model.config import CyberSLMConfig, default_config  # noqa: E402
from cyberslm.model.model import build_model, count_parameters    # noqa: E402


def load(ckpt: Path, device: torch.device):
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "config" in payload:
        cfg, state = CyberSLMConfig(**payload["config"]), payload["model_state"]
    elif isinstance(payload, dict) and "model_state" in payload:
        cfg, state = default_config(), payload["model_state"]
    else:
        cfg, state = default_config(), payload
    model = build_model(cfg, device=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, payload


def main() -> int:
    ap = argparse.ArgumentParser(description="CyberSLM base model (text continuation)")
    ap.add_argument("--prompt", "-p", default="SQL injection is")
    ap.add_argument("--interactive", "-i", action="store_true")
    ap.add_argument("--checkpoint", "-c", default=str(_HERE / "models" / "base.pt"))
    ap.add_argument("--tokenizer", default=str(_HERE / "tokenizer" / "tokenizer_output" / "tokenizer.model"))
    ap.add_argument("--max-new-tokens", "-m", type=int, default=120)
    ap.add_argument("--temperature", "-t", type=float, default=0.8,
                    help="0 = greedy/deterministic")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model-info", action="store_true")
    args = ap.parse_args()

    import sentencepiece as spm

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Weights are not stored in this repository. Get them from:", file=sys.stderr)
        print("  https://huggingface.co/sabari2005/cyberslm-base", file=sys.stderr)
        print("or pass --checkpoint /path/to/checkpoint.pt", file=sys.stderr)
        return 1
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    model, cfg, payload = load(ckpt, device)

    print(f"model  : {ckpt.name}  ({count_parameters(model)['total']:,} params)")
    print(f"context: {cfg.max_seq_len}   vocab: {cfg.vocab_size:,}   device: {device}")
    if isinstance(payload, dict) and payload.get("step"):
        print(f"trained: step {payload['step']}  val_loss {payload.get('val_loss'):.4f}")
    if args.model_info:
        return 0

    def run(text: str) -> None:
        ids = sp.encode(text, out_type=int)
        if sp.bos_id() >= 0:
            ids = [sp.bos_id()] + ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        out = model.generate(
            x, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_id=sp.eos_id() if sp.eos_id() >= 0 else None,
        )
        dt = time.perf_counter() - t0
        new = out[0, len(ids):].tolist()
        if sp.eos_id() in new:
            new = new[: new.index(sp.eos_id())]
        print("\n" + "-" * 66)
        # Decode prompt+continuation TOGETHER. Decoding the continuation
        # alone and concatenating drops the word-boundary marker on its
        # first token, gluing the halves ("...is" + "a common" -> "isa").
        prompt_ids = [t for t in ids if t != sp.bos_id()]
        print(sp.decode(prompt_ids + new))
        print("-" * 66)
        print(f"{len(new)} tokens in {dt:.2f}s ({len(new)/dt if dt else 0:.1f} tok/s)\n")

    if args.interactive:
        print("\nBase model - type the START of a sentence, it continues it.")
        print("('exit' to quit)")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye."); break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("Bye."); break
            run(q)
        return 0

    run(args.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
