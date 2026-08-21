"""
CyberSLM on Modal
=================
Serverless GPU training for the CyberSLM base model and its SFT stage.

Workspace: mrdarkcomputer

Layout inside the container
---------------------------
    /root/cyberslm, /root/cyberslm_sft, /root/Preprocessing_Pipeline   code
    /data                                                             inputs  (volume)
    /runs                                                             outputs (volume)

Typical flow
------------
    # 0. one-time: push the data
    python infra/upload_data.py

    # 1. cheap GPU sanity check (~1 minute of GPU time)
    modal run infra/modal_app.py::smoke

    # 2. the real run, detached so it survives your laptop sleeping
    modal run --detach infra/modal_app.py::pretrain

    # 3. watch it
    modal app logs <app-id>

    # 4. pull the result down
    python infra/download_model.py

Design notes
------------
* Checkpoints are written to a Volume and committed after every save, so a
  preempted or crashed run loses at most `save_every` steps.
* Every entry point auto-resumes from `<out>/latest.txt` if present, which makes
  a re-run idempotent rather than destructive.
* `smoke` exists specifically so a typo in a path or a shape bug surfaces after
  ~1 minute of billed GPU time instead of an hour into the real run.
"""

from __future__ import annotations

import modal

APP_NAME = "cyberslm"

# torch is deliberately unpinned: the CUDA build that PyPI serves for the
# image's Python version is the one we want, and pinning to a version that
# later disappears breaks the image build for no benefit. The version actually
# used is logged at the top of every run.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "sentencepiece", "psutil")
    .add_local_dir("cyberslm", "/root/cyberslm")
    .add_local_dir("cyberslm_sft", "/root/cyberslm_sft")
    .add_local_dir("Preprocessing_Pipeline", "/root/Preprocessing_Pipeline")
)

app = modal.App(APP_NAME, image=image)

# NOTE: a pre-existing "cyberslm-data" volume in this workspace belongs to a
# DIFFERENT project (~/Desktop/moe): vocab 32768, a MoE + recursion + memory
# architecture, and five phases of live checkpoints. These volumes are
# deliberately named apart from it so nothing here can overwrite that work.
data_vol = modal.Volume.from_name("cyberslm-v1-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("cyberslm-v1-runs", create_if_missing=True)

VOLUMES = {"/data": data_vol, "/runs": runs_vol}

# Measured, not assumed (see smoke_* below, 12 timed steps each at 131,072
# tokens/step, seq_len 2048):
#
#   A10G      batch=8  accum=8   1.939 s/step    67,596 tok/s   47% VRAM
#   A100-40   batch=16 accum=4   0.719 s/step   182,179 tok/s   51% VRAM
#
# The A100 is 2.7x the throughput for 1.9x the hourly rate, so it is both
# faster and cheaper per token. The larger micro-batch is what lets it stay
# busy; at batch=8 a card this size is mostly idle between kernel launches.
DEFAULT_GPU = "A100-40GB"
HOURS = 60 * 60


def _bootstrap():
    """Put the shipped repo code on sys.path and turn on INFO logging."""
    import logging
    import sys

    for p in ("/root", "/root/Preprocessing_Pipeline"):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Trainer/CheckpointManager report progress through logging.getLogger(),
    # and without a configured root handler Python silently drops everything
    # below WARNING. The first run looked frozen for exactly this reason: the
    # print() banner appeared and not one training step did. force=True
    # overrides any handler a library installed on import.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _banner(tag: str):
    import torch

    print("=" * 70, flush=True)
    print(f"  CyberSLM :: {tag}", flush=True)
    print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"  GPU: {props.name} | {props.total_memory / 1e9:.1f} GB "
            f"| bf16={torch.cuda.is_bf16_supported()}",
            flush=True,
        )
    print("=" * 70, flush=True)


# --------------------------------------------------------------------------- #
# Cheap pre-flight check
# --------------------------------------------------------------------------- #

def _run_smoke(batch_size: int = 8, accum_steps: int = 8, steps: int = 12,
               seq_len: int = 2048):
    """
    Prove the GPU path end to end for about a minute of billed time.

    Builds the real model at the real width, runs real optimizer steps against
    the real corpus, and measures STEADY-STATE throughput. Anything that would
    fail an hour into `pretrain` (missing data, shape bug, OOM, no bf16) fails
    here instead, for a couple of cents.
    """
    _bootstrap()
    _banner("smoke test")

    import time
    from pathlib import Path

    import torch
    from dataloader import make_train_dataloader, make_val_dataloader
    from cyberslm.model.config import CyberSLMConfig
    from cyberslm.model.model import count_parameters
    from cyberslm.training.config import TrainingConfig
    from cyberslm.training.trainer import Trainer

    for f in ("train.bin", "val.bin", "tokenizer.model"):
        p = Path("/data") / f
        if not p.exists():
            raise FileNotFoundError(f"/data/{f} missing - run infra/upload_data.py first")
        print(f"  {f}: OK ({p.stat().st_size / 1e6:.1f} MB)", flush=True)

    mc = CyberSLMConfig(max_seq_len=seq_len).validate()
    # grad_accum matches production. A small-accum sweep understates tok/s
    # because per-step fixed costs dominate when each step does few micro-batches.
    tc = TrainingConfig(
        seq_len=seq_len, batch_size=batch_size, grad_accum_steps=accum_steps,
        max_steps=steps, warmup_steps=2, val_every_steps=10_000, val_steps=2,
        save_every_steps=10_000, log_every_steps=1000,
        checkpoint_dir="/runs/_smoke", dtype="bfloat16",
    ).validate()

    tl, _ = make_train_dataloader(Path("/data/train.bin"), seq_len, batch_size,
                                  num_workers=4)
    vl = make_val_dataloader(Path("/data/val.bin"), seq_len, batch_size,
                             num_workers=2)
    print(f"  train windows: {len(tl.dataset):,} | val windows: {len(vl.dataset):,}",
          flush=True)
    print(f"  config: batch={batch_size} accum={accum_steps} seq={seq_len} "
          f"-> {batch_size * accum_steps * seq_len:,} tok/step", flush=True)

    trainer = Trainer(mc, tc, tl, vl)
    print(f"  params: {count_parameters(trainer.model)['total']:,}", flush=True)

    # Time each optimizer step. The first few include CUDA context creation,
    # allocator warmup and the first data fetch; averaging over them would
    # understate steady-state speed and skew the GPU choice.
    durations = []
    tokens_per_step = batch_size * accum_steps * seq_len
    orig_step = trainer.optimizer.step
    mark = {"t": None}

    def timed_step(*a, **kw):
        out = orig_step(*a, **kw)
        torch.cuda.synchronize()
        now = time.time()
        if mark["t"] is not None:
            durations.append(now - mark["t"])
        mark["t"] = now
        return out

    trainer.optimizer.step = timed_step
    mark["t"] = time.time()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    warm = durations[2:] or durations
    sec_step = sum(warm) / len(warm)
    steady = tokens_per_step / sec_step
    peak = torch.cuda.max_memory_allocated() / 1e9
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("")
    print(f"  steps timed     : {len(durations)} ({len(warm)} after warmup)", flush=True)
    print(f"  first step      : {durations[0]:.2f}s" if durations else "", flush=True)
    print(f"  steady-state    : {sec_step:.3f} s/step", flush=True)
    print(f"  THROUGHPUT      : {steady:,.0f} tok/s", flush=True)
    print(f"  peak GPU mem    : {peak:.2f} / {total_mem:.1f} GB "
          f"({100 * peak / total_mem:.0f}%)", flush=True)
    print(f"  wall (incl save): {wall:.1f}s", flush=True)

    # Projection for the real run, from the measured rate.
    for target in (6000,):
        secs = target * sec_step
        print(f"  -> {target} steps = {secs / 3600:.2f} h "
              f"({target * tokens_per_step / 1e6:.0f}M tokens)", flush=True)

    import shutil
    shutil.rmtree("/runs/_smoke", ignore_errors=True)
    runs_vol.commit()
    return {
        "tokens_per_sec": steady,
        "sec_per_step": sec_step,
        "peak_gb": peak,
        "total_gb": total_mem,
        "gpu": torch.cuda.get_device_properties(0).name,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
    }


# Per-GPU wrappers. Modal fixes the accelerator at decoration time, so probing a
# second card means a second function rather than a runtime argument. Each costs
# a couple of cents and the answer decides hours of billed training.

@app.function(gpu="A10G", volumes=VOLUMES, timeout=20 * 60)
def smoke(batch_size: int = 8, accum_steps: int = 8, steps: int = 12,
          seq_len: int = 2048):
    return _run_smoke(batch_size, accum_steps, steps, seq_len)


@app.function(gpu="A100-40GB", volumes=VOLUMES, timeout=20 * 60)
def smoke_a100(batch_size: int = 16, accum_steps: int = 4, steps: int = 12,
               seq_len: int = 2048):
    return _run_smoke(batch_size, accum_steps, steps, seq_len)


@app.function(gpu="L40S", volumes=VOLUMES, timeout=20 * 60)
def smoke_l40s(batch_size: int = 16, accum_steps: int = 4, steps: int = 12,
               seq_len: int = 2048):
    return _run_smoke(batch_size, accum_steps, steps, seq_len)


# --------------------------------------------------------------------------- #
# Pretraining
# --------------------------------------------------------------------------- #

@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=8 * HOURS)
def pretrain(
    steps: int = 6000,
    seq_len: int = 2048,
    batch_size: int = 16,
    accum_steps: int = 4,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup: int = 600,
    dtype: str = "bfloat16",
    run_name: str = "base",
    save_every: int = 250,
    val_every: int = 250,
    val_steps: int = 60,
    resume: bool = True,
):
    """Pretrain the base model, committing checkpoints to the runs volume."""
    _bootstrap()
    _banner(f"pretrain :: {run_name}")

    from pathlib import Path

    from dataloader import make_train_dataloader, make_val_dataloader
    from cyberslm.model.config import CyberSLMConfig
    from cyberslm.model.model import count_parameters
    from cyberslm.training.config import TrainingConfig
    from cyberslm.training.trainer import Trainer

    out = Path("/runs") / run_name
    out.mkdir(parents=True, exist_ok=True)

    resume_from = None
    if resume:
        latest = out / "latest.txt"
        if latest.exists():
            cand = Path(latest.read_text().strip())
            if cand.exists():
                resume_from = cand
                print(f"  resuming from {cand}", flush=True)

    mc = CyberSLMConfig(max_seq_len=seq_len).validate()
    tc = TrainingConfig(
        train_data_path="/data/train.bin", val_data_path="/data/val.bin",
        seq_len=seq_len, batch_size=batch_size, grad_accum_steps=accum_steps,
        max_steps=steps, warmup_steps=warmup, learning_rate=lr, min_lr=min_lr,
        dtype=dtype, val_every_steps=val_every, val_steps=val_steps,
        save_every_steps=save_every, log_every_steps=25, keep_last_n=3,
        checkpoint_dir=str(out),
    ).validate()

    tl, _ = make_train_dataloader(Path("/data/train.bin"), seq_len, batch_size,
                                  num_workers=4)
    vl = make_val_dataloader(Path("/data/val.bin"), seq_len, batch_size,
                             num_workers=2)

    n_tokens = Path("/data/train.bin").stat().st_size // 2
    per_step = batch_size * accum_steps * seq_len
    print(f"  corpus      : {n_tokens:,} tokens", flush=True)
    print(f"  tokens/step : {per_step:,}", flush=True)
    print(f"  total       : {per_step * steps:,} "
          f"({per_step * steps / n_tokens:.2f} epochs)", flush=True)
    print(f"  windows     : {len(tl.dataset):,} train / {len(vl.dataset):,} val",
          flush=True)

    trainer = Trainer(mc, tc, tl, vl, resume_from=resume_from)
    print(f"  params      : {count_parameters(trainer.model)['total']:,}", flush=True)

    # Commit the volume on every checkpoint write so a preemption costs at most
    # `save_every` steps rather than the whole run.
    _orig_save = trainer.ckpt_manager.save

    def _save_and_commit(*a, **kw):
        p = _orig_save(*a, **kw)
        runs_vol.commit()
        return p

    trainer.ckpt_manager.save = _save_and_commit

    trainer.train()
    runs_vol.commit()

    best = out / "best.pt"
    print(f"\n  DONE. best={best} exists={best.exists()} "
          f"size={best.stat().st_size / 1e6:.0f} MB" if best.exists() else "  DONE.",
          flush=True)
    return {"run": run_name, "best_val_loss": trainer.best_val_loss,
            "steps": trainer.global_step}


# --------------------------------------------------------------------------- #
# Supervised fine-tuning
# --------------------------------------------------------------------------- #

@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=6 * HOURS)
def sft(
    base_run: str = "base",
    run_name: str = "sft",
    epochs: int = 3,
    batch_size: int = 8,
    accum_steps: int = 4,
    lr: float = 2e-5,
    max_seq_len: int = 2048,
    dtype: str = "bfloat16",
):
    """Instruction-tune the pretrained base on /data/SFT.jsonl."""
    _bootstrap()
    _banner(f"sft :: {run_name} (base={base_run})")

    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, "/root/cyberslm_sft")

    from configs.sft_config import default_config
    from data.prompt_formatter import Tokenizer
    from data.sft_dataset import build_datasets
    from trainer import SFTTrainer
    from utils.seed import set_seed

    base_ckpt = Path("/runs") / base_run / "best.pt"
    if not base_ckpt.exists():
        raise FileNotFoundError(f"No base checkpoint at {base_ckpt}. Run pretrain first.")

    out = Path("/runs") / run_name
    out.mkdir(parents=True, exist_ok=True)

    cfg = default_config()
    cfg.tokenizer.model_path = "/data/tokenizer.model"
    cfg.data.train_path = "/data/SFT.jsonl"
    cfg.data.max_seq_len = max_seq_len
    cfg.data.num_workers = 2
    cfg.train.pretrained_checkpoint = str(base_ckpt)
    cfg.train.output_dir = str(out)
    cfg.train.num_epochs = epochs
    cfg.train.per_device_batch_size = batch_size
    cfg.train.gradient_accumulation_steps = accum_steps
    cfg.train.learning_rate = lr
    cfg.train.dtype = dtype
    cfg.train.run_name = run_name

    set_seed(cfg.train.seed)

    # Inherit the base model's exact architecture rather than trusting defaults.
    from train import _inherit_arch_from_checkpoint
    _inherit_arch_from_checkpoint(cfg)

    device = torch.device("cuda")
    from model.cyberslm import CyberSLM as SFTModel
    model = SFTModel(cfg.model).to(device)

    from utils.checkpoint_manager import CheckpointManager
    CheckpointManager.load_pretrained(str(base_ckpt), model, device, strict=True)

    tokenizer = Tokenizer(cfg.tokenizer.model_path)
    train_ds, val_ds = build_datasets(cfg, tokenizer)
    print(f"  SFT samples : {len(train_ds):,} train / {len(val_ds):,} val", flush=True)

    trainer = SFTTrainer(model, train_ds, val_ds, cfg, device, tokenizer)

    _orig = trainer._run_and_log_validation

    def _val_and_commit(*a, **kw):
        r = _orig(*a, **kw)
        runs_vol.commit()
        return r

    trainer._run_and_log_validation = _val_and_commit

    trainer.train()
    runs_vol.commit()
    print("\n  DONE.", flush=True)
    return {"run": run_name, "best_val_loss": trainer.state.best_val_loss}


# --------------------------------------------------------------------------- #
# Inspection helper
# --------------------------------------------------------------------------- #

@app.function(volumes=VOLUMES, timeout=10 * 60)
def ls():
    """List what is currently on both volumes (no GPU, effectively free)."""
    from pathlib import Path

    for root in ("/data", "/runs"):
        print(f"\n=== {root} ===", flush=True)
        base = Path(root)
        if not base.exists():
            print("  (missing)", flush=True)
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                print(f"  {p.stat().st_size / 1e6:10.1f} MB  {p.relative_to(base)}",
                      flush=True)


@app.local_entrypoint()
def main():
    print("Pick an entry point explicitly, e.g.:")
    print("  modal run infra/modal_app.py::smoke")
    print("  modal run --detach infra/modal_app.py::pretrain")
    print("  modal run --detach infra/modal_app.py::sft")
    print("  modal run infra/modal_app.py::ls")
