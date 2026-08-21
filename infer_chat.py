"""
Instruction-tuned model inference — question answering.

    python Final/infer_chat.py --prompt "What is SQL injection?"
    python Final/infer_chat.py --interactive

The prompt is built with the SAME formatter used during fine-tuning, so the
model sees exactly the token sequence it was trained on. Hand-assembling the
prompt string instead produces different token ids at every segment boundary
(SentencePiece prepends a word-start marker per encode call) and the model then
sees something it was never trained on.

Expect correctly-shaped answers with unreliable facts: this is a 33.5M-parameter
model. See runs/reports/FINAL_REPORT.md for measured behaviour.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "cyberslm_sft"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from configs.sft_config import default_config                      # noqa: E402
from data.prompt_formatter import PromptFormatter, Tokenizer       # noqa: E402
from model.cyberslm import CyberSLM as SFTModel                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="CyberSLM instruct (question answering)")
    ap.add_argument("--prompt", "-p", default="What is SQL injection and how do I prevent it?")
    ap.add_argument("--interactive", "-i", action="store_true")
    ap.add_argument("--checkpoint", "-c", default=str(_HERE / "models" / "instruct.pt"))
    ap.add_argument("--tokenizer", default=str(_HERE / "tokenizer" / "tokenizer_output" / "tokenizer.model"))
    ap.add_argument("--max-new-tokens", "-m", type=int, default=200)
    ap.add_argument("--temperature", "-t", type=float, default=0.0,
                    help="0 = greedy/deterministic (recommended for this model)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Weights are not stored in this repository. Get them from:", file=sys.stderr)
        print("  https://huggingface.co/sabari2005/cyberslm-instruct", file=sys.stderr)
        print("or pass --checkpoint /path/to/checkpoint.pt", file=sys.stderr)
        return 1

    cfg = default_config()
    cfg.tokenizer.model_path = args.tokenizer
    cfg.model.max_seq_len = 2048
    cfg.data.max_seq_len = 2048

    tok = Tokenizer(cfg.tokenizer.model_path)
    fmt = PromptFormatter(cfg=cfg, tokenizer=tok)

    model = SFTModel(cfg.model)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device).eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"model  : {ckpt.name}  ({n:,} params)")
    print(f"context: {cfg.model.max_seq_len}   device: {device}   "
          f"decoding: {'greedy' if args.temperature == 0 else f'T={args.temperature}'}")

    def answer(question: str) -> None:
        ids = fmt.format_for_inference({"messages": [{"role": "user", "content": question}]})
        x = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        out = model.generate(
            x, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, eos_id=tok.eos_id,
        )
        dt = time.perf_counter() - t0
        new = out[0, len(ids):].tolist()
        stopped = tok.eos_id in new
        if stopped:
            new = new[: new.index(tok.eos_id)]
        print("\n" + "-" * 66)
        print(tok.decode(new).strip() or "(empty)")
        print("-" * 66)
        print(f"{len(new)} tokens in {dt:.2f}s ({len(new)/dt if dt else 0:.1f} tok/s), "
              f"{'stopped on EOS' if stopped else 'hit token limit'}\n")

    if args.interactive:
        print("\nInstruct model - ask a question. ('exit' to quit)")
        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye."); break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("Bye."); break
            answer(q)
        return 0

    answer(args.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
