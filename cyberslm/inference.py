"""
CyberSLM Inference
==================
Text generation for the pretrained (base) CyberSLM model.

    python cyberslm/inference.py --prompt "In cybersecurity, a firewall is"
    python cyberslm/inference.py --interactive
    python cyberslm/inference.py --model-info

What changed
------------
* Sampling and the decode loop now live in ``CyberSLM.generate()``, which uses a
  KV cache. The old loop here re-ran the full 12-layer stack over the entire
  prefix for every token (O(n^2)); it is now O(n).
* Checkpoint discovery no longer hardcodes ``cyberslm/checkpoints/best.pt``.
  Training writes to ``runs/<name>/`` relative to the repo root, so the two
  disagreed and inference could not find what training had just produced.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import sentencepiece as spm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cyberslm.model.config import CyberSLMConfig, default_config   # noqa: E402
from cyberslm.model.model import CyberSLM, build_model, model_summary  # noqa: E402

# Searched in order; first hit wins.
_CKPT_CANDIDATES = [
    _REPO_ROOT / "runs" / "base" / "best.pt",
    _REPO_ROOT / "runs" / "base" / "latest.pt",
    _REPO_ROOT / "cyberslm" / "checkpoints" / "best.pt",   # legacy layout
    _REPO_ROOT / "checkpoints" / "best.pt",                # legacy layout
]
_DEFAULT_TOKENIZER = _REPO_ROOT / "tokenizer" / "tokenizer_output" / "tokenizer.model"


def find_checkpoint(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            for name in ("best.pt", "latest.pt"):
                if (p / name).exists():
                    return p / name
            raise FileNotFoundError(f"No best.pt/latest.pt inside {p}")
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    for c in _CKPT_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "No checkpoint found. Looked in:\n  "
        + "\n  ".join(str(c) for c in _CKPT_CANDIDATES)
        + "\nPass --checkpoint explicitly, or train first."
    )


def load_tokenizer(path: Path) -> spm.SentencePieceProcessor:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(path))
    return sp


def load_model(checkpoint_path: Path, device: torch.device) -> CyberSLM:
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    step = payload.get("step", "?") if isinstance(payload, dict) else "?"
    vl = payload.get("val_loss") if isinstance(payload, dict) else None
    print(f"  step={step}  val_loss={f'{vl:.4f}' if vl is not None else 'n/a'}  device={device}")
    return model


@torch.inference_mode()
def generate_text(
    model: CyberSLM,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: torch.device,
) -> tuple[str, int]:
    ids: List[int] = sp.encode(prompt, out_type=int)
    bos = sp.bos_id()
    if bos is not None and bos >= 0:
        ids = [bos] + ids

    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=sp.eos_id() if sp.eos_id() >= 0 else None,
    )
    new_ids = out[0, len(ids):].tolist()
    if sp.eos_id() in new_ids:
        new_ids = new_ids[: new_ids.index(sp.eos_id())]
    return sp.decode(new_ids), len(new_ids)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CyberSLM base-model text generation")
    p.add_argument("--prompt", "-p", default="In cybersecurity, a firewall is")
    p.add_argument("--interactive", "-i", action="store_true")
    p.add_argument("--max-new-tokens", "-m", type=int, default=256)
    p.add_argument("--temperature", "-t", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--checkpoint", "-c", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--model-info", action="store_true")
    p.add_argument("--device", default=None)
    return p


def _run(model, sp, prompt, args, device) -> None:
    print("\n" + "-" * 60 + f"\nPROMPT: {prompt}\n" + "-" * 60)
    t0 = time.perf_counter()
    text, n = generate_text(
        model, sp, prompt, args.max_new_tokens, args.temperature,
        args.top_k, args.top_p, args.repetition_penalty, device,
    )
    dt = time.perf_counter() - t0
    print(text)
    print("-" * 60)
    print(f"{n} tokens in {dt:.2f}s ({n / dt if dt > 0 else 0:.1f} tok/s)\n")


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sp = load_tokenizer(Path(args.tokenizer) if args.tokenizer else _DEFAULT_TOKENIZER)
    print(f"Tokenizer vocab: {sp.get_piece_size():,}")
    model = load_model(find_checkpoint(args.checkpoint), device)

    if args.model_info:
        print(model_summary(model))
        return 0

    if args.interactive:
        print("\nInteractive mode (type 'exit' to quit)")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye."); break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("Bye."); break
            _run(model, sp, q, args, device)
        return 0

    _run(model, sp, args.prompt, args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
