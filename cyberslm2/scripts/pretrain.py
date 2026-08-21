"""
pretrain.py
===========
Stage 1: pretrain the base model on the packed token corpus.

    # inspect the plan without touching a GPU (safe on any laptop)
    python -m cyberslm2.scripts.pretrain --dry-run

    # real run
    python -m cyberslm2.scripts.pretrain --preset base-50m

--dry-run prints the full training plan (parameter count, token budget, memory
estimate, schedule) and exits before allocating anything. Use it to sanity-check
a configuration on the machine you *write* code on, then launch the real run on
the machine with the GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cyberslm2.configs.presets import MODEL_PRESETS, TrainConfig, get_model_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain CyberSLM-2")
    p.add_argument("--preset", default="base-50m", choices=sorted(MODEL_PRESETS))
    p.add_argument("--train-bin", default="tokenizer/data/train.bin")
    p.add_argument("--val-bin", default="tokenizer/data/val.bin")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Muon LR for hidden matrices")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--checkpoint-dir", default="cyberslm2/checkpoints/pretrain")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume", default=None, help="path to a checkpoint .pt")
    p.add_argument("--device", default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without allocating or training")
    return p


def make_configs(args):
    mcfg = get_model_config(args.preset)
    mcfg.max_seq_len = args.seq_len

    tcfg = TrainConfig(
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        optimizer=args.optimizer,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        compile_model=not args.no_compile,
    )
    if args.max_steps is not None:
        tcfg.max_steps = args.max_steps
    if args.lr is not None:
        tcfg.lr = args.lr
    return mcfg, tcfg


def print_plan(mcfg, tcfg, args) -> None:
    p = mcfg.param_count()
    corpus_tokens = 0
    train_path = Path(tcfg.train_bin)
    if train_path.exists():
        corpus_tokens = train_path.stat().st_size // 2

    # rough training memory: params + grads + optimizer state, all fp32-ish
    n = p["total"]
    bytes_per_param = 4 + 4 + (4 if tcfg.optimizer == "muon" else 8)
    state_gb = n * bytes_per_param / 1e9
    act_gb = (
        tcfg.micro_batch_size * tcfg.seq_len * mcfg.d_model
        * mcfg.n_layers * 2 * 2 / 1e9
    )

    print("=" * 70)
    print(f"CyberSLM-2 pretraining plan  [{args.preset}]")
    print("=" * 70)
    print(f"  parameters       {p['total']:,} ({p['total']/1e6:.2f}M), "
          f"non-emb {p['non_embedding']/1e6:.2f}M")
    print(f"  layers/d_model   {mcfg.n_layers} / {mcfg.d_model}")
    print(f"  heads (q/kv)     {mcfg.n_heads} / {mcfg.n_kv_heads}")
    print(f"  context          {mcfg.max_seq_len}")
    print(f"  optimizer        {tcfg.optimizer}  (lr={tcfg.lr}, "
          f"adamw_lr={tcfg.adamw_lr})")
    print(f"  schedule         {tcfg.schedule}  warmup={tcfg.warmup_steps} "
          f"decay_frac={tcfg.decay_frac}")
    print(f"  tokens/step      {tcfg.tokens_per_step():,}")
    print(f"  total tokens     {tcfg.total_tokens():,} "
          f"({tcfg.total_tokens()/1e9:.2f}B) over {tcfg.max_steps:,} steps")
    if corpus_tokens:
        print(f"  corpus           {corpus_tokens:,} tokens "
              f"-> {tcfg.total_tokens()/corpus_tokens:.1f} epochs")
    else:
        print(f"  corpus           NOT FOUND at {tcfg.train_bin}")
    print(f"  est. state mem   ~{state_gb:.2f} GB (params+grads+optimizer)")
    print(f"  est. activations ~{act_gb:.2f} GB at micro_batch="
          f"{tcfg.micro_batch_size}")
    print(f"  doc-aware mask   {tcfg.doc_aware_mask}")
    print("=" * 70)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    mcfg, tcfg = make_configs(args)
    print_plan(mcfg, tcfg, args)

    if args.dry_run:
        print("\n--dry-run: nothing was allocated and no training was started.")
        return 0

    # Imports deferred so --dry-run works even without torch installed.
    import torch
    from torch.utils.data import DataLoader

    from cyberslm2.data.datasets import PackedPretrainDataset
    from cyberslm2.model.transformer import CyberSLM2
    from cyberslm2.training.trainer import Trainer

    if not Path(tcfg.train_bin).exists():
        print(f"\nERROR: {tcfg.train_bin} not found. Run the preprocessing "
              f"pipeline first (see README Quickstart).")
        return 1

    train_ds = PackedPretrainDataset(tcfg.train_bin, tcfg.seq_len, seed=tcfg.seed)
    val_ds = (
        PackedPretrainDataset(tcfg.val_bin, tcfg.seq_len, seed=tcfg.seed, shuffle=False)
        if Path(tcfg.val_bin).exists() else None
    )
    logging.info("train windows: %d | val windows: %d",
                 len(train_ds), len(val_ds) if val_ds else 0)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=tcfg.micro_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=tcfg.micro_batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=pin, drop_last=False,
                   persistent_workers=args.num_workers > 0)
        if val_ds else None
    )

    model = CyberSLM2(mcfg)
    print(model.summary())

    if tcfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logging.info("torch.compile enabled")
        except Exception as exc:
            logging.warning("torch.compile unavailable (%s); continuing eager", exc)

    trainer = Trainer(model, mcfg, tcfg, train_loader, val_loader,
                      device=args.device, mode="pretrain")
    if args.resume:
        trainer.load(args.resume)

    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
