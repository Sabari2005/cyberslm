"""
harness.py
==========
Capability evaluation.

Two scoring modes, because they measure different things:

  loglikelihood (multiple choice)
      Score each candidate answer by the model's total log-probability of its
      tokens given the prompt, then pick the highest. Length-normalized by token
      count, otherwise the shortest option wins every time purely because it has
      fewer negative log-probs to accumulate.

      This mode needs no generation, so it is stable and cheap -- the right tool
      for MMLU-style knowledge and for a security-specific question bank.

  generative (exact match / contains / tool-call validity)
      The model writes an answer and it is compared to a reference. This is the
      only way to measure whether the model can actually *produce* a well-formed
      tool call, a working command, or a correct arithmetic result.

Task files are JSONL. Multiple choice:
    {"question": "...", "choices": ["a","b","c","d"], "answer": 0}
Generative:
    {"prompt": "...", "answer": "...", "mode": "contains"}
Tool calling:
    {"prompt": "...", "expect_tool": "nmap_scan"}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from cyberslm2.data.special_tokens import (
    ASSISTANT_ID,
    BOS_ID,
    END_ID,
    EOS_ID,
    STOP_IDS,
    TOOL_CALL_ID,
    TOOL_END_ID,
    USER_ID,
)


@dataclass
class TaskResult:
    name: str
    n: int
    correct: int
    metric: str = "accuracy"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def __str__(self) -> str:
        return f"{self.name:<28} {self.score:6.2%}  ({self.correct}/{self.n})"


def _chat_prompt(sp, user_text: str) -> list[int]:
    """Wrap a bare question in the training-time chat format."""
    return (
        [BOS_ID, USER_ID]
        + sp.encode(user_text, out_type=int)
        + [END_ID, ASSISTANT_ID]
    )


@torch.no_grad()
def sequence_logprob(
    model,
    prompt_ids: list[int],
    continuation_ids: list[int],
    device,
) -> tuple[float, int]:
    """
    Total log P(continuation | prompt) and the continuation's token count.

    Only positions inside the continuation are scored. The logit at index i
    predicts token i+1, so the continuation's first token is predicted by the
    logit at the prompt's last position.
    """
    ids = torch.tensor([prompt_ids + continuation_ids], device=device)
    logits, _ = model(ids)
    logprobs = F.log_softmax(logits.float(), dim=-1)

    start = len(prompt_ids) - 1
    total = 0.0
    for j, tok in enumerate(continuation_ids):
        total += logprobs[0, start + j, tok].item()
    return total, len(continuation_ids)


@torch.no_grad()
def eval_multiple_choice(
    model, sp, path: Path, device, limit: Optional[int] = None,
    length_normalize: bool = True,
) -> TaskResult:
    model.eval()
    correct = n = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        prompt = _chat_prompt(sp, item["question"])

        scores = []
        for choice in item["choices"]:
            cont = sp.encode(str(choice), out_type=int)
            if not cont:
                scores.append(-math.inf)
                continue
            lp, ntok = sequence_logprob(model, prompt, cont, device)
            scores.append(lp / ntok if length_normalize else lp)

        if int(scores.index(max(scores))) == int(item["answer"]):
            correct += 1
        n += 1
        if limit and n >= limit:
            break

    return TaskResult(path.stem, n, correct, "accuracy")


@torch.no_grad()
def eval_generative(
    model, sp, path: Path, device, limit: Optional[int] = None,
    max_new_tokens: int = 256,
) -> TaskResult:
    model.eval()
    correct = n = 0
    samples: list[dict[str, str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        prompt = _chat_prompt(sp, item["prompt"])
        ids = torch.tensor([prompt], device=device)

        out = model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=0.0,
            stop_ids=STOP_IDS,
        )
        text = sp.decode(out[0, len(prompt):].tolist()).strip()

        mode = item.get("mode", "contains")
        ref = str(item.get("answer", "")).strip()
        if mode == "exact":
            hit = text == ref
        elif mode == "tool":
            hit = _valid_tool_call(out[0, len(prompt):].tolist(), sp,
                                   item.get("expect_tool"))
        else:
            hit = ref.lower() in text.lower()

        correct += bool(hit)
        n += 1
        if len(samples) < 3:
            samples.append({"prompt": item["prompt"][:100], "got": text[:160]})
        if limit and n >= limit:
            break

    return TaskResult(path.stem, n, correct, "match", {"samples": samples})


def _valid_tool_call(ids: list[int], sp, expect_tool: Optional[str]) -> bool:
    """
    A tool call counts only if it is syntactically well formed.

    That means: an opening <|tool_call|>, a closing <|/tool|>, valid JSON in
    between with a "name" field, and the expected tool name. Partial credit for
    "looks roughly like JSON" would hide exactly the failures that break a real
    agent loop.
    """
    if TOOL_CALL_ID not in ids or TOOL_END_ID not in ids:
        return False
    start = ids.index(TOOL_CALL_ID) + 1
    end = ids.index(TOOL_END_ID)
    if end <= start:
        return False
    try:
        payload = json.loads(sp.decode(ids[start:end]).strip())
    except (json.JSONDecodeError, ValueError):
        return False
    if "name" not in payload:
        return False
    return expect_tool is None or payload["name"] == expect_tool


@torch.no_grad()
def eval_perplexity(model, loader, device, max_batches: int = 100) -> TaskResult:
    """Token-weighted perplexity on a held-out set."""
    from cyberslm2.data.packing import build_doc_causal_mask, build_doc_ids

    model.eval()
    total_nll, total_tok = 0.0, 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        tokens = batch.to(device) if torch.is_tensor(batch) else batch["input_ids"].to(device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        mask = build_doc_causal_mask(build_doc_ids(x))
        logits, _ = model(x, attn_mask=mask)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tok += y.numel()

    ppl = math.exp(total_nll / total_tok) if total_tok else float("inf")
    return TaskResult("perplexity", total_tok, 0, "ppl", {"ppl": ppl})


def run_suite(
    model, sp, task_dir: str | Path, device, limit: Optional[int] = None,
) -> list[TaskResult]:
    """
    Run every task file in a directory.

    Naming convention drives the scorer: files starting with 'mc_' are scored by
    loglikelihood, 'gen_' by generation.
    """
    task_dir = Path(task_dir)
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    results = []
    for path in sorted(task_dir.glob("*.jsonl")):
        if path.name.startswith("mc_"):
            results.append(eval_multiple_choice(model, sp, path, device, limit))
        elif path.name.startswith("gen_"):
            results.append(eval_generative(model, sp, path, device, limit))
    return results


def print_report(results: list[TaskResult]) -> None:
    print("\n" + "=" * 58)
    print("EVALUATION REPORT")
    print("=" * 58)
    for r in results:
        if r.metric == "ppl":
            print(f"{r.name:<28} {r.extra['ppl']:8.3f}  ({r.n:,} tokens)")
        else:
            print(str(r))

    scored = [r for r in results if r.metric != "ppl" and r.n]
    if scored:
        macro = sum(r.score for r in scored) / len(scored)
        print("-" * 58)
        print(f"{'macro average':<28} {macro:6.2%}")
    print("=" * 58)
