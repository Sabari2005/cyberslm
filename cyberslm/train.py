"""
CyberSLM Pretraining Entry Point
================================

Usage
-----
    python cyberslm/train.py --help
    python cyberslm/train.py --steps 4000 --seq-len 2048 --batch-size 8
    python cyberslm/train.py --resume runs/base/latest.txt

Note
----
This file previously had no argument parser at all, so `python train.py --help`
silently ignored the flag and started a real multi-hour training run. Every
knob is now an explicit flag and nothing starts until `main()` is called with
parsed arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "Preprocessing_Pipeline") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "Preprocessing_Pipeline"))

from dataloader import make_train_dataloader, make_val_dataloader  # noqa: E402
from cyberslm.model.config import CyberSLMConfig                    # noqa: E402
from cyberslm.training.config import TrainingConfig                 # noqa: E402
from cyberslm.training.trainer import Trainer                       # noqa: E402

log = logging.getLogger("pretrain")
BYTES_PER_TOKEN = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pretrain CyberSLM on a flat uint16 token stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = p.add_argument_group("data")
    d.add_argument("--train-bin", default=str(_REPO_ROOT / "tokenizer" / "data" / "train.bin"))
    d.add_argument("--val-bin",   default=str(_REPO_ROOT / "tokenizer" / "data" / "val.bin"))
    d.add_argument("--num-workers", type=int, default=4)

    m = p.add_argument_group("model")
    m.add_argument("--vocab-size", type=int, default=32000)
    m.add_argument("--seq-len", type=int, default=2048,
                   help="Context length. Also sets the model's max_seq_len, so "
                        "every RoPE position the model will ever use is trained.")
    m.add_argument("--hidden-dim", type=int, default=384)
    m.add_argument("--layers", type=int, default=12)
    m.add_argument("--heads", type=int, default=6)
    m.add_argument("--ffn-dim", type=int, default=1024)

    t = p.add_argument_group("training")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch-size", type=int, default=8, help="micro-batch per device")
    t.add_argument("--accum-steps", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--min-lr", type=float, default=3e-5)
    t.add_argument("--warmup", type=int, default=None, help="default: 10%% of --steps")
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--compile", action="store_true", help="enable torch.compile")

    o = p.add_argument_group("output")
    o.add_argument("--out-dir", default=str(_REPO_ROOT / "runs" / "base"))
    o.add_argument("--val-every", type=int, default=200)
    o.add_argument("--val-steps", type=int, default=100)
    o.add_argument("--save-every", type=int, default=200)
    o.add_argument("--log-every", type=int, default=10)
    o.add_argument("--keep-last", type=int, default=3)
    o.add_argument("--resume", default=None, help="path to a .pt checkpoint")

    p.add_argument("--dry-run", action="store_true",
                   help="Build everything, print the plan, and exit WITHOUT training.")
    return p


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    train_bin, val_bin = Path(args.train_bin), Path(args.val_bin)
    for pth in (train_bin, val_bin):
        if not pth.exists():
            log.error("Missing data file: %s", pth)
            log.error("Run Preprocessing_Pipeline/dataset_tokenizer.py then dataset_builder.py.")
            return 1

    model_config = CyberSLMConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        head_dim=args.hidden_dim // args.heads,
        ffn_hidden_dim=args.ffn_dim,
    ).validate()

    train_config = TrainingConfig(
        train_data_path=str(train_bin),
        val_data_path=str(val_bin),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.accum_steps,
        max_steps=args.steps,
        warmup_steps=args.warmup if args.warmup is not None else max(1, args.steps // 10),
        learning_rate=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dtype=args.dtype,
        val_every_steps=args.val_every,
        val_steps=args.val_steps,
        save_every_steps=args.save_every,
        log_every_steps=args.log_every,
        keep_last_n=args.keep_last,
        checkpoint_dir=args.out_dir,
        seed=args.seed,
        compile_model=args.compile,
    ).validate()

    n_tokens = train_bin.stat().st_size // BYTES_PER_TOKEN
    tokens_per_step = args.batch_size * args.accum_steps * args.seq_len
    total_tokens = tokens_per_step * args.steps
    log.info("Corpus      : %s tokens (%s windows @ %d)",
             f"{n_tokens:,}", f"{(n_tokens - 1) // args.seq_len:,}", args.seq_len)
    log.info("Tokens/step : %s   Total: %s   Epochs: %.2f",
             f"{tokens_per_step:,}", f"{total_tokens:,}", total_tokens / n_tokens)
    log.info("Output dir  : %s", args.out_dir)

    train_loader, _ = make_train_dataloader(
        bin_path=train_bin, context_len=args.seq_len,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )
    val_loader = make_val_dataloader(
        bin_path=val_bin, context_len=args.seq_len,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    log.info("Batches/epoch: %s train | %s val", f"{len(train_loader):,}", f"{len(val_loader):,}")

    if args.dry_run:
        log.info("--dry-run: configuration is valid. Exiting without training.")
        return 0

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model_config=model_config,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main(build_parser().parse_args()))
