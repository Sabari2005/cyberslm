"""
Open SLM Leaderboard benchmarks on Modal.

    modal run --detach infra/modal_bench.py::bench_all

Runs the four tasks the leaderboard ranks on, for both published models:

    HellaSwag, ARC-Easy, ARC-Challenge, PIQA   via lm-evaluation-harness
    ArithMark-3, ArithMark-2                   via AxiomicLabs' own script

Everything is 0-shot. These models are 33.5M parameters with a 2048 context;
few-shot prompting mostly consumes context without helping at this scale, and
0-shot is what the leaderboard's existing sub-10M entries are consistent with
(their HellaSwag/PIQA scores sit at chance, which few-shot would not produce).
The n-shot setting is stated explicitly in the submission so it is reproducible.

acc_norm is reported alongside acc for every task. The leaderboard's own
ArithMark script defaults to `--primary-metric acc_norm`, and acc_norm is the
convention for these multiple-choice tasks, so acc_norm is what gets submitted.
"""

from __future__ import annotations

import modal

app = modal.App("cyberslm-bench")

# lm-eval pulls a large dependency tree; pin nothing so the resolver picks a
# consistent set, and log the exact versions in the results for reproducibility.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "sentencepiece", "protobuf",
        "datasets", "accelerate", "huggingface_hub", "tqdm",
    )
    # Install the harness from its canonical source. `pip install "lm-eval[hf]"`
    # completed without error but left no lm_eval module -- pip skips unknown
    # extras with only a warning, so the failure was silent until the GPU
    # function crashed on import.
    .run_commands(
        "pip install --no-cache-dir "
        "git+https://github.com/EleutherAI/lm-evaluation-harness.git",
        # Fail the IMAGE BUILD, not a billed GPU run, if the import is broken.
        "python -c \"import lm_eval; print('lm_eval', lm_eval.__version__)\"",
    )
    .add_local_dir("bench", "/root/bench")
)

results_vol = modal.Volume.from_name("cyberslm-bench-results", create_if_missing=True)
VOLUMES = {"/results": results_vol}

MODELS = {
    "base": "sabari2005/cyberslm-base",
    "instruct": "sabari2005/cyberslm-instruct",
}
LM_EVAL_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa"]


def _log_versions() -> dict:
    import torch, transformers
    info = {"torch": torch.__version__, "transformers": transformers.__version__}
    try:
        import lm_eval
        info["lm_eval"] = lm_eval.__version__
    except Exception:
        info["lm_eval"] = "unknown"
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_properties(0).name
    print("  versions:", info, flush=True)
    return info


@app.function(gpu="A10G", volumes=VOLUMES, timeout=4 * 60 * 60)
def run_lm_eval(model_id: str, tag: str, tasks: list[str] = None, num_fewshot: int = 0):
    """Run lm-evaluation-harness on one model."""
    import json
    from pathlib import Path

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or LM_EVAL_TASKS
    print("=" * 70, flush=True)
    print(f"  lm-eval :: {model_id}  ({num_fewshot}-shot)", flush=True)
    print("=" * 70, flush=True)
    versions = _log_versions()

    lm = HFLM(pretrained=model_id, batch_size=32, dtype="float32",
              max_length=2048, trust_remote_code=False)
    out = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=num_fewshot, batch_size=32,
    )

    scores = {}
    for task, r in out["results"].items():
        scores[task] = {k: v for k, v in r.items()
                        if isinstance(v, (int, float)) and not k.endswith("_stderr,none")}
        print(f"  {task:<16} " + "  ".join(
            f"{k}={v:.4f}" for k, v in scores[task].items() if "acc" in k), flush=True)

    blob = {"model": model_id, "tag": tag, "num_fewshot": num_fewshot,
            "versions": versions, "results": scores}
    Path("/results").mkdir(exist_ok=True)
    Path(f"/results/lm_eval_{tag}.json").write_text(json.dumps(blob, indent=2))
    results_vol.commit()
    return blob


@app.function(gpu="A10G", volumes=VOLUMES, timeout=2 * 60 * 60)
def run_arithmark(model_id: str, tag: str, version: str = "3"):
    """Run the official ArithMark script (unmodified) on one model."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    print("=" * 70, flush=True)
    print(f"  ArithMark-{version} :: {model_id}", flush=True)
    print("=" * 70, flush=True)
    _log_versions()

    # The two ArithMark releases do not share a naming convention.
    names = {
        "3": ("arithmark3/bencharithmark-3.py", "arithmark3/arithmark-3.jsonl"),
        "2": ("arithmark2/benchmark_arithmark-2.0.py", "arithmark2/arithmark_2.0.jsonl"),
    }
    rel_script, rel_data = names[version]
    script = Path("/root/bench") / rel_script
    data = Path("/root/bench") / rel_data
    if not script.exists():
        raise FileNotFoundError(f"missing {script}")

    out_dir = Path(f"/results/arithmark{version}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--model", model_id,
        "--data-path", str(data),
        "--device", "cuda",
        "--dtype", "float32",          # batch-size-stable reference scoring
        "--batch-size", "32",
        "--primary-metric", "acc_norm",
        "--results-dir", str(out_dir),
    ]
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90 * 60)
    print(proc.stdout[-6000:], flush=True)
    if proc.returncode != 0:
        print("STDERR:\n" + proc.stderr[-4000:], flush=True)
        raise RuntimeError(f"ArithMark-{version} failed rc={proc.returncode}")

    found = {}
    for f in out_dir.rglob("*.json"):
        try:
            found[f.name] = json.loads(f.read_text())
        except Exception:
            pass
    results_vol.commit()
    return {"model": model_id, "tag": tag, "version": version,
            "stdout_tail": proc.stdout[-3000:], "files": found}


@app.function(volumes=VOLUMES, timeout=30 * 60)
def summarise():
    """Collect everything on the results volume and compute the Intelligence Index."""
    import json
    from pathlib import Path

    def chance_norm(score: float, chance: float) -> float:
        """(score - chance) / (100 - chance), clamped at 0. Leaderboard convention."""
        return max(0.0, (score - chance) / (100.0 - chance)) * 100.0

    out = {}
    for f in sorted(Path("/results").rglob("*.json")):
        try:
            out[str(f.relative_to("/results"))] = json.loads(f.read_text())
        except Exception:
            pass
    print(json.dumps(out, indent=2)[:8000], flush=True)
    return out


@app.local_entrypoint()
def bench_all():
    print("Benchmarking both models for the Open SLM Leaderboard\n")
    for tag, mid in MODELS.items():
        print(f"=== {tag}: {mid} ===")
        r = run_lm_eval.remote(mid, tag)
        print(json.dumps(r["results"], indent=2)[:1500])
        for v in ("3", "2"):
            try:
                a = run_arithmark.remote(mid, tag, v)
                print(a["stdout_tail"][-1200:])
            except Exception as exc:
                print(f"ArithMark-{v} failed for {tag}: {exc}")


import json  # noqa: E402  (used by the local entrypoint)
