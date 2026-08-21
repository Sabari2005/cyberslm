"""
Benchmark Hugging Face causal language models exclusively on ArithMark 2.0.

The benchmark data is downloaded from the official dataset repo when needed:
https://huggingface.co/datasets/AxiomicLabs/Arithmark-2.0
"""

import argparse
import json
import urllib.request
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


OFFICIAL_REPO = "https://huggingface.co/datasets/AxiomicLabs/Arithmark-2.0"
DATA_URL = f"{OFFICIAL_REPO}/resolve/main/arithmark_2.0.jsonl"
DATA_FILE = "arithmark_2.0.jsonl"
CACHE_DIR = "benchmark_cache"
BATCH_SIZE = 16
MAX_CONTEXT = 1024

# (model_name,) or (model_name, tokenizer_name)
MODELS = [
    ("AxiomicLabs/GPT-X2-125M",),
    ("harley-ml/dillion-1.2M",),
    ("HuggingFaceTB/SmolLM-135M",),
]


def ensure_arithmark_data(data_path: Path, force_download: bool = False) -> Path:
    """Download the official ArithMark 2.0 JSONL file if it is not present."""
    if data_path.exists() and not force_download:
        return data_path

    data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ArithMark 2.0 data to {data_path}...")
    urllib.request.urlretrieve(DATA_URL, data_path)
    return data_path


def load_arithmark_2(data_path: Path):
    examples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(
                {
                    "ctx": item["ctx"],
                    "endings": item["endings"],
                    "label": int(item["label"]),
                    "metadata": item.get("metadata", {}),
                }
            )
    return examples


def load_hf_model(model_name: str, tokenizer_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    return model, tokenizer


def tokenize_request(tokenizer, context: str, continuation: str):
    ctx_tokens = tokenizer(context, add_special_tokens=False).input_ids
    # ArithMark 2.0 continuations already include the leading space documented
    # by the official dataset, so score the raw continuation as provided.
    cont_tokens = tokenizer(continuation, add_special_tokens=False).input_ids
    tokens = ctx_tokens + cont_tokens

    if len(tokens) > MAX_CONTEXT:
        tokens = tokens[-MAX_CONTEXT:]
        ctx_len = max(1, len(tokens) - len(cont_tokens))
    else:
        ctx_len = len(ctx_tokens)

    return tokens, max(0, len(tokens) - ctx_len)


def evaluate_arithmark_2(model, tokenizer, device: str, examples, batch_size: int):
    correct = 0
    total = 0
    grouped = {}

    for idx_start in tqdm(range(0, len(examples), batch_size), desc="  arithmark_2.0"):
        batch_ex = examples[idx_start:idx_start + batch_size]
        batch_tokens = []
        batch_cont_lens = []
        ex_offsets = []

        for ex in batch_ex:
            flat_start = len(batch_tokens)
            for ending in ex["endings"]:
                tokens, cont_len = tokenize_request(tokenizer, ex["ctx"], ending)
                batch_tokens.append(tokens)
                batch_cont_lens.append(cont_len)
            ex_offsets.append((flat_start, len(ex["endings"])))

        max_len = max(len(tokens) for tokens in batch_tokens)
        padded = [tokens + [tokenizer.pad_token_id] * (max_len - len(tokens)) for tokens in batch_tokens]
        tokens_t = torch.tensor(padded, dtype=torch.long, device=device)
        lengths = torch.tensor([len(tokens) for tokens in batch_tokens], device=device)
        attention_mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]

        attention_mask = attention_mask.bool()

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )

        with torch.no_grad():
            with autocast_context:
                logits = model(tokens_t, attention_mask=attention_mask).logits

        log_probs = F.log_softmax(logits.float(), dim=-1)

        for ex_idx, ex in enumerate(batch_ex):
            flat_start, num_choices = ex_offsets[ex_idx]
            lls = []
            for choice_idx in range(num_choices):
                flat_idx = flat_start + choice_idx
                tokens_i = batch_tokens[flat_idx]
                cont_len = batch_cont_lens[flat_idx]
                start = len(tokens_i) - cont_len
                ll = 0.0
                for pos in range(start, len(tokens_i)):
                    if pos > 0:
                        ll += log_probs[flat_idx, pos - 1, tokens_i[pos]].item()
                lls.append(ll)

            pred = max(range(num_choices), key=lambda i: lls[i])
            label = int(ex["label"])
            correct += int(pred == label)
            total += 1

            operator_count = ex.get("metadata", {}).get("operator_count", "unknown")
            if operator_count not in grouped:
                grouped[operator_count] = [0, 0]
            grouped[operator_count][0] += int(pred == label)
            grouped[operator_count][1] += 1

        del tokens_t, logits, log_probs
        if device == "cuda":
            torch.cuda.empty_cache()

    acc = correct / total * 100 if total else 0.0
    print(f"  arithmark_2.0: acc {acc:.2f}% ({correct}/{total})")

    if grouped:
        groups = sorted(grouped, key=lambda value: (str(type(value)), value))
        header = "  " + "  ".join(f"ops={group!s:>3}" for group in groups) + f"  {'Avg':>6}"
        vals = []
        for group in groups:
            group_correct, group_total = grouped[group]
            vals.append(group_correct / group_total * 100 if group_total else 0.0)
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        print("  " + "  ".join(f"{value:>6.2f}%" for value in vals) + f"  {acc:>5.2f}%")

    return {"acc": acc, "correct": correct, "total": total}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ArithMark 2.0 only.")
    parser.add_argument("--model", action="append", help="HF model id. Can be passed multiple times.")
    parser.add_argument("--tokenizer", help="Tokenizer id to use for all --model entries.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data-path", default=DATA_FILE, help="Path for arithmark_2.0.jsonl. Downloaded if missing.")
    parser.add_argument("--force-download", action="store_true", help="Redownload arithmark_2.0.jsonl even if it exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_path = ensure_arithmark_data(Path(args.data_path).resolve(), force_download=args.force_download)
    examples = load_arithmark_2(data_path)
    print(f"Loaded {len(examples)} ArithMark 2.0 examples from {data_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_entries = [(model,) for model in args.model] if args.model else MODELS
    all_results = {}

    for model_entry in model_entries:
        model_name = model_entry[0]
        tokenizer_name = args.tokenizer or (model_entry[1] if len(model_entry) > 1 else model_name)

        print(f"\n{'=' * 60}")
        print(f"  Loading {model_name}...")
        print(f"{'=' * 60}")

        try:
            model, tokenizer = load_hf_model(model_name, tokenizer_name, device)
        except Exception as exc:
            print(f"  Failed to load model: {exc}")
            continue

        total_params = sum(param.numel() for param in model.parameters())
        print(f"  {total_params:,} parameters")

        result = evaluate_arithmark_2(model, tokenizer, device, examples, args.batch_size)
        all_results[model_name] = result

        print(f"\n{'=' * 60}")
        print(f"  {model_name} ({total_params:,} params) RESULTS")
        print(f"{'=' * 60}")
        print(f"  arithmark_2.0 {result['acc']:>9.2f}%")
        print(f"{'=' * 60}")

        model_tag = model_name.replace("/", "_")
        results_dir = Path(CACHE_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{model_tag}_arithmark_2.0_results.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model_name,
                    "params": total_params,
                    "dataset": str(data_path),
                    "results": {"arithmark_2.0": result},
                },
                f,
                indent=2,
            )
        print(f"Results saved to {results_file}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  FINAL SUMMARY")
        print(f"{'=' * 60}")
        for model_name, result in all_results.items():
            print(f"  {model_name:<45} {result['acc']:>7.2f}%")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
