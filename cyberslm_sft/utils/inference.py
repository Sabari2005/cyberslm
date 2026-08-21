"""
CyberSLM SFT — Inference Tests
================================
Runs greedy / top-p generation on a fixed set of sanity-check prompts
after training completes.  These are NOT evaluation metrics — they are
qualitative checks that the model produces coherent, instruction-following
output.

Prompts are chosen to cover the primary SFT objectives:
* Explain a security concept (SQL Injection, AES, CIA Triad)
* Define a broad topic (Linux)
* Generate code (Python)
* Summarise text

The output is written to both the logger and an optional text file so it
can be reviewed offline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.sft_config import SFTConfig
from data.prompt_formatter import PromptFormatter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default sanity-check prompts (alpaca format)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: List[dict] = [
    {
        "instruction": "Explain SQL Injection.",
        "input":       "",
    },
    {
        "instruction": "What is AES?",
        "input":       "",
    },
    {
        "instruction": "Write a Python function to reverse a string.",
        "input":       "",
    },
    {
        "instruction": "Explain the CIA Triad.",
        "input":       "",
    },
    {
        "instruction": "What is Linux?",
        "input":       "",
    },
    {
        "instruction": "Summarize the following text.",
        "input": (
            "Cryptography is the practice and study of techniques for secure "
            "communication in the presence of adversarial behaviour. Modern "
            "cryptography exists at the intersection of mathematics, computer "
            "science, and electrical engineering."
        ),
    },
]


# ---------------------------------------------------------------------------
# Token sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model:          nn.Module,
    input_ids:      torch.Tensor,
    tokenizer,
    max_new_tokens: int   = 256,
    temperature:    float = 0.7,
    top_p:          float = 0.9,
    eos_id:         int   = 3,
    device:         torch.device = torch.device("cpu"),
) -> str:
    """
    Autoregressive generation with temperature scaling and nucleus sampling.

    Parameters
    ----------
    model:          The instruction-tuned model in eval mode.
    input_ids:      Prompt token ids, shape ``(1, T_prompt)``.
    tokenizer:      Tokenizer with a ``.decode()`` method.
    max_new_tokens: Maximum tokens to generate.
    temperature:    Softmax temperature (1.0 = unscaled, <1.0 = sharper).
    top_p:          Nucleus sampling probability threshold.
    eos_id:         Token id that signals end of generation.
    device:         Device for inference.

    Returns
    -------
    Decoded response string (generated tokens only, prompt stripped).
    """
    model.eval()
    ids = input_ids.to(device)

    generated: List[int] = []

    for _ in range(max_new_tokens):
        # Forward pass — take last T tokens if exceeds context
        context = ids[:, -model_context_len(model):]
        try:
            logits = model(context)
        except TypeError:
            logits = model(context, attention_mask=None)

        if isinstance(logits, torch.Tensor):
            next_token_logits = logits[0, -1, :]          # (V,)
        elif hasattr(logits, "logits"):
            next_token_logits = logits.logits[0, -1, :]
        else:
            raise TypeError(f"Unexpected model output type: {type(logits)}")

        # Temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Nucleus (top-p) sampling
        next_id = _top_p_sample(next_token_logits, top_p)

        if next_id == eos_id:
            break

        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated)


def _top_p_sample(logits: torch.Tensor, top_p: float) -> int:
    """Sample from the nucleus (top-p fraction of probability mass)."""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative probability exceeds top_p
    sorted_probs[cumulative - sorted_probs > top_p] = 0.0

    # Renormalise and sample
    sorted_probs /= sorted_probs.sum()
    sample_idx = torch.multinomial(sorted_probs, num_samples=1).item()
    return int(sorted_idx[sample_idx].item())


def model_context_len(model: nn.Module) -> int:
    """
    Return the model's maximum context length.
    Looks for ``model.max_seq_len`` or ``model.config.max_seq_len`` attributes.
    Falls back to 4096.
    """
    if hasattr(model, "max_seq_len"):
        return model.max_seq_len
    if hasattr(model, "config") and hasattr(model.config, "max_seq_len"):
        return model.config.max_seq_len
    return 4096


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_inference_tests(
    model:     nn.Module,
    tokenizer,
    cfg:       SFTConfig,
    device:    torch.device,
    prompts:   Optional[List[dict]] = None,
    save_path: Optional[str]        = None,
) -> List[dict]:
    """
    Run inference on the sanity-check prompt set and log results.

    Parameters
    ----------
    model:      Instruction-tuned model (will be set to eval mode).
    tokenizer:  Initialised tokenizer.
    cfg:        Master config (for generation hyperparameters).
    device:     Inference device.
    prompts:    Override default prompts (list of ``{instruction, input}`` dicts).
    save_path:  If provided, write results to this text file.

    Returns
    -------
    List of result dicts ``{instruction, input, response, elapsed_s}``.
    """
    prompts = prompts or DEFAULT_PROMPTS
    # Build prompts with the SAME segment encoder used in training.
    # ConversationTemplate renders one string that is then encoded in a
    # single call; training encodes each segment separately. With
    # add_dummy_prefix=True those two produce DIFFERENT token ids at every
    # segment boundary, so the model was being prompted off-distribution.
    formatter = PromptFormatter(cfg=cfg, tokenizer=tokenizer)
    results: List[dict] = []

    logger.info("")
    logger.info("=" * 60)
    logger.info("INFERENCE SANITY CHECKS (%d prompts)", len(prompts))
    logger.info("=" * 60)

    model.eval()
    torch.set_grad_enabled(False)

    for i, prompt_sample in enumerate(prompts, start=1):
        # Build inference prompt (no response appended)
        sample_for_inference = {
            "instruction": prompt_sample["instruction"],
            "input":       prompt_sample.get("input", ""),
            "output":      "",   # placeholder; not used in inference mode
        }
        # format_for_inference() prepends BOS and emits the "### Response:"
        # header without any response body -- byte-identical to the prompt half
        # of a training example.
        input_ids = formatter.format_for_inference(sample_for_inference)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        t0 = time.time()
        response = generate(
            model=model,
            input_ids=input_tensor,
            tokenizer=tokenizer,
            max_new_tokens=cfg.train.max_new_tokens,
            temperature=cfg.train.temperature,
            top_p=cfg.train.top_p,
            eos_id=tokenizer.eos_id,
            device=device,
        )
        elapsed = time.time() - t0

        result = {
            "prompt_index": i,
            "instruction":  prompt_sample["instruction"],
            "input":        prompt_sample.get("input", ""),
            "response":     response,
            "elapsed_s":    round(elapsed, 2),
        }
        results.append(result)

        # Pretty print
        logger.info("")
        logger.info("─" * 50)
        logger.info("[%d/%d] %s", i, len(prompts), prompt_sample["instruction"])
        if prompt_sample.get("input"):
            logger.info("Input: %s ...", prompt_sample["input"][:80])
        logger.info("Response (%ds):", int(elapsed))
        for line in response.strip().split("\n"):
            logger.info("  %s", line)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Inference tests complete.")
    logger.info("=" * 60)

    # Optional file save
    if save_path:
        _write_results(results, save_path)

    torch.set_grad_enabled(True)
    model.train()

    return results


def _write_results(results: List[dict], path: str) -> None:
    """Write inference results to a plain text file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("CyberSLM-Instruct Inference Test Results\n")
        fh.write("=" * 60 + "\n\n")
        for r in results:
            fh.write(f"[{r['prompt_index']}] Instruction: {r['instruction']}\n")
            if r["input"]:
                fh.write(f"    Input: {r['input'][:120]}\n")
            fh.write(f"    Response ({r['elapsed_s']}s):\n")
            for line in r["response"].strip().split("\n"):
                fh.write(f"        {line}\n")
            fh.write("\n")
    logger.info("Inference results written to %s", path)
