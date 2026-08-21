"""Benchmark Hugging Face causal language models on ArithMark 3.0."""

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATA_FILE = "arithmark-3.jsonl"
CACHE_DIR = "benchmark_cache"
BATCH_SIZE = 32
MAX_CONTEXT = 1024

# Some Hub repositories use an absolute import for a Python file that lives
# beside their modeling file. Transformers can otherwise mistake that file for
# a missing PyPI package before loading the repository's remote code.
REMOTE_CODE_SIBLING_IMPORTS = {
    "liodon-ai/slm-10m": ("configuration_slm.py",),
}

# (model_name,) or (model_name, tokenizer_name)
MODELS = [
    ("AxiomicLabs/GPT-X2-125M",),
    ("harley-ml/dillion-1.2M",),
    ("HuggingFaceTB/SmolLM2-135M",),
]


def load_arithmark_3(data_path: Path):
    """Load and validate ArithMark 3.0 continuation examples."""
    examples = []
    with data_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{data_path}:{line_number}: invalid JSON") from exc

            context = item.get("ctx")
            endings = item.get("endings")
            label = item.get("label")
            category = item.get("activity_label", "unknown")
            example_id = item.get("ind", f"line_{line_number}")

            if isinstance(label, str):
                try:
                    label = int(label)
                except ValueError as exc:
                    raise ValueError(
                        f"{data_path}:{line_number}: label must be an integer"
                    ) from exc

            if not isinstance(context, str) or not context:
                raise ValueError(
                    f"{data_path}:{line_number}: ctx must be a non-empty string"
                )
            if not isinstance(endings, list) or len(endings) < 2:
                raise ValueError(
                    f"{data_path}:{line_number}: endings must contain at least two strings"
                )
            if not all(isinstance(ending, str) and ending for ending in endings):
                raise ValueError(
                    f"{data_path}:{line_number}: every ending must be a non-empty string"
                )
            if not all(ending.startswith(" ") for ending in endings):
                raise ValueError(
                    f"{data_path}:{line_number}: every continuation must retain its leading space"
                )
            if type(label) is not int or not 0 <= label < len(endings):
                raise ValueError(
                    f"{data_path}:{line_number}: invalid label {label!r}"
                )

            examples.append(
                {
                    "id": str(example_id),
                    "ctx": context,
                    "endings": endings,
                    "label": label,
                    "category": str(category),
                }
            )

    if not examples:
        raise ValueError(f"No examples found in {data_path}")
    return examples


def resolve_device(requested: str):
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def remote_code_sibling_imports(model_name: str, enabled: bool):
    """Temporarily expose known remote-code sibling modules as imports."""
    filenames = REMOTE_CODE_SIBLING_IMPORTS.get(model_name, ()) if enabled else ()
    added_paths = []
    previously_loaded = set(sys.modules)

    try:
        for filename in filenames:
            module_file = Path(
                hf_hub_download(repo_id=model_name, filename=filename)
            )
            module_directory = str(module_file.parent)
            if module_directory not in sys.path:
                sys.path.insert(0, module_directory)
                added_paths.append(module_directory)
        yield
    finally:
        for module_directory in added_paths:
            if module_directory in sys.path:
                sys.path.remove(module_directory)
        for filename in filenames:
            module_name = Path(filename).stem
            if module_name not in previously_loaded:
                sys.modules.pop(module_name, None)


def load_hf_model(
    model_name: str,
    tokenizer_name: str,
    device: str,
    trust_remote_code: bool,
    attn_implementation: str,
    dtype_name: str,
):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_name]
    if device == "cpu" and dtype != torch.float32:
        print(f"  Warning: {dtype_name} requested on CPU; using float32 instead")
        dtype = torch.float32
    if (
        dtype == torch.bfloat16
        and device == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("This CUDA device does not support bfloat16")

    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    with remote_code_sibling_imports(model_name, trust_remote_code):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token_id = 0

    return model, tokenizer, dtype


def tokenize_request(
    tokenizer,
    context: str,
    continuation: str,
    max_context: int,
):
    """Tokenize context and continuation independently."""
    ctx_tokens = tokenizer(context, add_special_tokens=False).input_ids
    cont_tokens = tokenizer(continuation, add_special_tokens=False).input_ids

    if not ctx_tokens:
        raise ValueError("A context tokenized to zero tokens")
    if not cont_tokens:
        raise ValueError("A continuation tokenized to zero tokens")
    if len(cont_tokens) >= max_context:
        raise ValueError(
            f"Continuation has {len(cont_tokens)} tokens, leaving no scoring "
            f"context under --max-context {max_context}"
        )

    available_context = max_context - len(cont_tokens)
    ctx_tokens = ctx_tokens[-available_context:]
    return ctx_tokens + cont_tokens, len(cont_tokens)


def prepare_arithmark_requests(tokenizer, examples, max_context: int):
    """Tokenize once and retain each example's continuation requests."""
    prepared = []
    for example in tqdm(examples, desc="  tokenizing", leave=False):
        requests = [
            tokenize_request(
                tokenizer,
                example["ctx"],
                ending,
                max_context,
            )
            for ending in example["endings"]
        ]
        prepared.append(
            {
                "example": example,
                "requests": requests,
                "max_length": max(len(tokens) for tokens, _ in requests),
            }
        )

    # Similar lengths together sharply reduce padding and full-vocabulary logits.
    prepared.sort(key=lambda item: item["max_length"])
    return prepared


def evaluate_arithmark_3(
    model,
    tokenizer,
    device: str,
    dtype,
    examples,
    batch_size: int,
    max_context: int,
    save_predictions: bool,
):
    import time

    prepare_start = time.perf_counter()
    prepared = prepare_arithmark_requests(tokenizer, examples, max_context)
    preparation_seconds = time.perf_counter() - prepare_start

    raw_correct = 0
    norm_correct = 0
    total = 0
    grouped = defaultdict(
        lambda: {"raw_correct": 0, "norm_correct": 0, "total": 0}
    )
    predictions = []
    evaluation_start = time.perf_counter()

    for batch_start in tqdm(
        range(0, len(prepared), batch_size),
        desc="  arithmark-3",
    ):
        batch_items = prepared[batch_start : batch_start + batch_size]
        batch_tokens = []
        continuation_lengths = []
        example_offsets = []

        for item in batch_items:
            flat_start = len(batch_tokens)
            for tokens, continuation_length in item["requests"]:
                batch_tokens.append(tokens)
                continuation_lengths.append(continuation_length)
            example_offsets.append(
                (flat_start, len(item["example"]["endings"]))
            )

        maximum_length = max(len(tokens) for tokens in batch_tokens)
        padded = [
            tokens + [tokenizer.pad_token_id] * (maximum_length - len(tokens))
            for tokens in batch_tokens
        ]
        tokens_t = torch.tensor(padded, dtype=torch.long, device=device)
        lengths = torch.tensor(
            [len(tokens) for tokens in batch_tokens],
            dtype=torch.long,
            device=device,
        )
        continuation_starts = torch.tensor(
            [
                len(tokens) - continuation_length
                for tokens, continuation_length in zip(
                    batch_tokens,
                    continuation_lengths,
                )
            ],
            dtype=torch.long,
            device=device,
        )
        attention_mask = (
            torch.arange(maximum_length, device=device)[None, :]
            < lengths[:, None]
        )

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            logits = model(
                tokens_t,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

            target_positions = torch.arange(
                1,
                maximum_length,
                device=device,
            )[None, :]
            score_mask = (
                (target_positions >= continuation_starts[:, None])
                & (target_positions < lengths[:, None])
            )
            request_indices, relative_positions = score_mask.nonzero(
                as_tuple=True
            )
            token_positions = relative_positions + 1

            selected_logits = logits[
                request_indices,
                token_positions - 1,
                :,
            ]
            selected_targets = tokens_t[request_indices, token_positions]
            token_log_likelihoods = -F.cross_entropy(
                selected_logits.float(),
                selected_targets,
                reduction="none",
            )

            request_count = len(batch_tokens)
            raw_scores_t = torch.zeros(
                request_count,
                dtype=torch.float32,
                device=device,
            )
            raw_scores_t.scatter_add_(
                0,
                request_indices,
                token_log_likelihoods,
            )
            token_counts_t = torch.zeros(
                request_count,
                dtype=torch.float32,
                device=device,
            )
            token_counts_t.scatter_add_(
                0,
                request_indices,
                torch.ones_like(token_log_likelihoods),
            )
            normalized_scores_t = raw_scores_t / token_counts_t.clamp_min(1)

        raw_scores_flat = raw_scores_t.cpu().tolist()
        normalized_scores_flat = normalized_scores_t.cpu().tolist()

        for item_index, item in enumerate(batch_items):
            example = item["example"]
            flat_start, number_of_choices = example_offsets[item_index]
            flat_end = flat_start + number_of_choices
            raw_scores = raw_scores_flat[flat_start:flat_end]
            normalized_scores = normalized_scores_flat[flat_start:flat_end]

            raw_prediction = max(
                range(number_of_choices),
                key=raw_scores.__getitem__,
            )
            norm_prediction = max(
                range(number_of_choices),
                key=normalized_scores.__getitem__,
            )
            label = example["label"]
            raw_hit = int(raw_prediction == label)
            norm_hit = int(norm_prediction == label)

            raw_correct += raw_hit
            norm_correct += norm_hit
            total += 1

            category = example["category"]
            grouped[category]["raw_correct"] += raw_hit
            grouped[category]["norm_correct"] += norm_hit
            grouped[category]["total"] += 1

            if save_predictions:
                predictions.append(
                    {
                        "id": example["id"],
                        "category": category,
                        "label": label,
                        "raw_prediction": raw_prediction,
                        "normalized_prediction": norm_prediction,
                        "raw_scores": raw_scores,
                        "normalized_scores": normalized_scores,
                    }
                )

        del tokens_t, logits, selected_logits, selected_targets

    evaluation_seconds = time.perf_counter() - evaluation_start
    raw_accuracy = raw_correct / total * 100 if total else 0.0
    normalized_accuracy = norm_correct / total * 100 if total else 0.0
    throughput = total / evaluation_seconds if evaluation_seconds else 0.0
    print(
        f"  arithmark-3: raw {raw_accuracy:.2f}% ({raw_correct}/{total})  "
        f"normalized {normalized_accuracy:.2f}% ({norm_correct}/{total})"
    )
    print(
        f"  speed: {throughput:.1f} examples/s  "
        f"(tokenize {preparation_seconds:.2f}s, "
        f"evaluate {evaluation_seconds:.2f}s)"
    )

    category_results = {}
    if grouped:
        print()
        print(f"  {'Category':<64} {'N':>5} {'Raw':>9} {'Normalized':>12}")
        print(f"  {'-' * 94}")
        for category in sorted(grouped):
            values = grouped[category]
            category_total = values["total"]
            raw_acc = values["raw_correct"] / category_total * 100
            norm_acc = values["norm_correct"] / category_total * 100
            category_results[category] = {
                "acc": raw_acc,
                "acc_norm": norm_acc,
                "raw_correct": values["raw_correct"],
                "norm_correct": values["norm_correct"],
                "total": category_total,
            }
            print(
                f"  {category:<64} {category_total:>5} "
                f"{raw_acc:>8.2f}% {norm_acc:>11.2f}%"
            )

    result = {
        "acc": raw_accuracy,
        "acc_norm": normalized_accuracy,
        "raw_correct": raw_correct,
        "norm_correct": norm_correct,
        "total": total,
        "categories": category_results,
        "timing": {
            "tokenization_seconds": preparation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "examples_per_second": throughput,
        },
    }
    if save_predictions:
        predictions.sort(key=lambda item: item["id"])
        result["predictions"] = predictions
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Hugging Face causal language models on ArithMark 3.0."
    )
    parser.add_argument(
        "--model", action="append", help="HF model ID; may be passed multiple times"
    )
    parser.add_argument(
        "--tokenizer", help="Tokenizer ID to use for every --model entry"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument(
        "--data-path",
        default=DATA_FILE,
        help="Path to the ArithMark 3.0 JSONL file",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--primary-metric",
        choices=("acc", "acc_norm"),
        default="acc_norm",
        help="Metric shown in model summaries; both metrics are always calculated",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help=(
            "bfloat16 is the fast default; float32 is available for "
            "batch-size-stable reference scoring"
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
        help=(
            "Optional Transformers attention backend; sdpa is usually "
            "fastest without extra packages"
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Use torch.compile(dynamic=True, mode='reduce-overhead'); "
            "best for repeated or larger runs"
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include per-question scores and predictions in the result JSON",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code for model and tokenizer loading",
    )
    parser.add_argument("--results-dir", default=CACHE_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_context < 2:
        raise SystemExit("--max-context must be at least 2")
    if args.dtype in ("float16", "bfloat16") and args.batch_size != BATCH_SIZE:
        print(
            "Warning: reduced-precision scores can depend on batch shape; "
            f"the standard configuration uses --batch-size {BATCH_SIZE}."
        )

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        raise SystemExit(f"ArithMark 3.0 data not found at {data_path}.")
    examples = load_arithmark_3(data_path)
    dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    print(f"Loaded {len(examples)} ArithMark 3.0 examples from {data_path}")
    print(f"ArithMark 3.0 dataset SHA-256: {dataset_sha256}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model_entries = [(model,) for model in args.model] if args.model else MODELS
    all_results = {}
    skipped_models = []

    for model_entry in model_entries:
        model_name = model_entry[0]
        tokenizer_name = (
            args.tokenizer
            or (model_entry[1] if len(model_entry) > 1 else model_name)
        )

        print(f"\n{'=' * 68}")
        print(f"  Loading {model_name}...")
        print(f"{'=' * 68}")

        try:
            model, tokenizer, dtype = load_hf_model(
                model_name,
                tokenizer_name,
                device,
                trust_remote_code=not args.no_trust_remote_code,
                attn_implementation=args.attn_implementation,
                dtype_name=args.dtype,
            )
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"  Failed to load model; skipping: {reason}")
            skipped_models.append((model_name, f"load: {reason}"))
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        total_params = sum(param.numel() for param in model.parameters())
        print(f"  {total_params:,} parameters ({dtype})")

        try:
            if args.compile:
                print(
                    "  Compiling model "
                    "(first batch will include compilation time)..."
                )
                model = torch.compile(
                    model,
                    mode="reduce-overhead",
                    dynamic=True,
                )

            result = evaluate_arithmark_3(
                model,
                tokenizer,
                device,
                dtype,
                examples,
                args.batch_size,
                args.max_context,
                args.save_predictions,
            )
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            if "out of memory" in reason.lower():
                print(
                    "  CUDA out of memory; skipping model. A smaller "
                    "--batch-size may work (the sequence batch is four "
                    "times that value)."
                )
            else:
                print(
                    "  Model failed during compilation/evaluation; "
                    f"skipping: {reason}"
                )
            skipped_models.append((model_name, f"evaluation: {reason}"))
            del model
            del tokenizer
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        result["primary_metric"] = args.primary_metric
        result["primary_acc"] = result[args.primary_metric]
        all_results[model_name] = result

        print(f"\n{'=' * 68}")
        print(f"  {model_name} ({total_params:,} params) RESULTS")
        print(f"{'=' * 68}")
        print(f"  Raw continuation accuracy     {result['acc']:>8.2f}%")
        print(
            f"  Length-normalized accuracy    "
            f"{result['acc_norm']:>8.2f}%"
        )
        print(
            f"  Primary ({args.primary_metric})"
            f"{result[args.primary_metric]:>16.2f}%"
        )
        print(f"{'=' * 68}")

        model_tag = model_name.replace("/", "_")
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = (
            results_dir / f"{model_tag}_arithmark-3_results.json"
        )
        with results_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": model_name,
                    "tokenizer": tokenizer_name,
                    "params": total_params,
                    "dataset": {
                        "path": str(data_path),
                        "sha256": dataset_sha256,
                        "items": len(examples),
                    },
                    "runtime": {
                        "device": device,
                        "dtype": str(dtype),
                        "batch_size": args.batch_size,
                        "max_context": args.max_context,
                        "attention_implementation": args.attn_implementation,
                        "compiled": args.compile,
                        "torch_version": torch.__version__,
                    },
                    "primary_metric": args.primary_metric,
                    "results": {"arithmark-3": result},
                },
                handle,
                indent=2,
            )
        print(f"Results saved to {results_file}")

        del model
        del tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        print(f"\n{'=' * 68}")
        print(f"  FINAL SUMMARY ({args.primary_metric})")
        print(f"{'=' * 68}")
        for model_name, result in all_results.items():
            print(
                f"  {model_name:<49} "
                f"arithmark-3: {result[args.primary_metric]:.2f}%"
            )
        print(f"{'=' * 68}")

    if skipped_models:
        print(f"\nSkipped {len(skipped_models)} model(s):")
        for model_name, reason in skipped_models:
            print(f"  {model_name}: {reason}")


if __name__ == "__main__":
    main()
