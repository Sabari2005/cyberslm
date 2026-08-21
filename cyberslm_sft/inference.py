"""
CyberSLM SFT Inference Engine
=============================
Production-quality text generation script for the CyberSLM fine-tuned model.
Supports interactive mode, streaming output, top-k/top-p sampling, and
repetition penalties.

Applies the SAME segment-based prompt encoding used during training so that
your inputs are correctly formatted for the instruct model.
"""

from __future__ import annotations

# --- path bootstrap: allow running this file from any working directory ---
import sys as _sys
from pathlib import Path as _Path
_SFT_ROOT = _Path(__file__).resolve().parent
if str(_SFT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SFT_ROOT))
# -------------------------------------------------------------------------

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from configs.sft_config import load_config
from data.prompt_formatter import PromptFormatter, Tokenizer
from model.cyberslm import CyberSLM

# ===========================================================================
# Sampling helpers
# ===========================================================================

def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    threshold = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_to_remove] = float("-inf")
    return sorted_logits.scatter(0, sorted_idx, sorted_logits)

def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    seen = torch.tensor(list(set(generated_ids)), dtype=torch.long)
    logits[seen] = logits[seen] / penalty
    return logits

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.3,
    generated_ids: list = [],
) -> int:
    logits = _apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    logits = _top_k_filter(logits, top_k)
    logits = _top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

# ===========================================================================
# Core generation function
# ===========================================================================

@torch.inference_mode()
def generate(
    model: CyberSLM,
    tokenizer: Tokenizer,
    formatter: PromptFormatter,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.3,
    device: torch.device = torch.device("cpu"),
    stream: bool = True,
) -> str:
    model.eval()

    # Format the prompt using the chat template.
    # We must include an empty assistant message at the end so the template
    # renders the `### Assistant:\n` header, which cues the model to respond.
    # Encode through the SAME segment path training used. Encoding a single
    # rendered string instead (the old behaviour) yields different token ids at
    # every segment boundary because SentencePiece prepends a dummy space
    # marker per encode() call.
    sample = {"messages": [{"role": "user", "content": prompt}]}
    token_ids = formatter.format_for_inference(sample)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    max_ctx = model.config.max_seq_len
    eos_id = tokenizer.eos_id

    generated_ids: List[int] = []
    
    # We use the underlying spm processor for streaming decoding
    sp_processor = tokenizer._sp
    prompt_tokens_no_bos = token_ids.copy()
    if len(prompt_tokens_no_bos) > 0 and prompt_tokens_no_bos[0] == tokenizer.bos_id:
        prompt_tokens_no_bos = prompt_tokens_no_bos[1:]

    _base = sp_processor.decode(prompt_tokens_no_bos)
    prev_len = len(_base)

    for _ in range(max_new_tokens):
        x_cond = x[:, -max_ctx:] if x.size(1) > max_ctx else x
        
        # SFT Model returns logits. We index the last token position
        logits = model(x_cond)
        if hasattr(logits, "logits"):
            logits = logits.logits
        next_logits = logits[0, -1, :]

        next_id = _sample_next_token(
            next_logits, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids,
        )

        if eos_id is not None and next_id == eos_id:
            break

        generated_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

        if stream:
            full_so_far = sp_processor.decode(prompt_tokens_no_bos + generated_ids)
            new_text = full_so_far[prev_len:]
            print(new_text, end="", flush=True)
            prev_len = len(full_so_far)

    if stream:
        print()

    return sp_processor.decode(generated_ids)

# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with your fine-tuned CyberSLM")
    
    # Resolve default paths relative to this script so it works from any CWD
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _DEFAULT_CHECKPOINT = _SCRIPT_DIR / "checkpoints" / "best"
    
    parser.add_argument("--prompt", "-p", type=str, default="What is a SQL injection attack?", help="Your message to the model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch an interactive REPL")
    parser.add_argument("--checkpoint", "-c", type=str, default=str(_DEFAULT_CHECKPOINT), help="Path to checkpoint directory")
    parser.add_argument("--max-new-tokens", "-m", type=int, default=512)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser

def _run_once(model, tokenizer, formatter, prompt, args, device):
    max_new = min(args.max_new_tokens, model.config.max_seq_len - 1)
    stream = not args.no_stream

    print("\n" + "─" * 60)
    print(f"USER: {prompt}")
    print("─" * 60 + "\nCyberSLM: ", end="", flush=True)

    t0 = time.perf_counter()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        formatter=formatter,
        prompt=prompt,
        max_new_tokens=max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
        stream=stream,
    )
    elapsed = time.perf_counter() - t0

    if not stream:
        print(output)

    # Calculate tokens per second
    gen_tokens = len(tokenizer.encode(output))
    tps = gen_tokens / elapsed if elapsed > 0 else 0

    print(f"\n{'─' * 60}")
    print(f"Generated {gen_tokens} tokens in {elapsed:.2f}s  ({tps:.1f} tok/s)")
    print("─" * 60 + "\n")

def main():
    parser = build_parser()
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.exists():
        print(f"Error: Checkpoint not found at {ckpt_dir}")
        sys.exit(1)

    print(f"Loading config from {ckpt_dir}...")
    cfg = load_config(ckpt_dir / "sft_config.json")
    
    print("Loading tokenizer...")
    # The config might contain absolute paths from the training server (e.g., Linux paths).
    # If the path doesn't exist on the local machine, fall back to a relative path.
    tok_path = Path(cfg.tokenizer.model_path)
    if not tok_path.exists():
        tok_path = Path(__file__).resolve().parent / "tokenizer" / "tokenizer.model"
    tokenizer = Tokenizer(str(tok_path))
    formatter = PromptFormatter(cfg=cfg, tokenizer=tokenizer)
    
    print(f"Loading model on {args.device}...")
    device = torch.device(args.device)
    model = CyberSLM(cfg.model).to(device)
    
    sd = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    if args.interactive:
        print("\nCyberSLM Interactive Mode  (type 'exit' or Ctrl-C to quit)")
        print("─" * 60)
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
            _run_once(model, tokenizer, formatter, prompt, args, device)
        return

    _run_once(model, tokenizer, formatter, args.prompt, args, device)

if __name__ == "__main__":
    main()
