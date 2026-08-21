"""
sft.py
======
Stage 2: supervised fine-tuning on instructions, reasoning traces, tool calls.

    python -m cyberslm2.scripts.sft --dry-run
    python -m cyberslm2.scripts.sft --pretrained cyberslm2/checkpoints/pretrain/best.pt

Loss is computed only on assistant-generated tokens. Prompt, system text and
tool *results* are context the model conditions on but is never asked to
predict -- training on them teaches the model to hallucinate tool output, which
is the single most damaging failure mode for an agentic model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cyberslm2.configs.presets import MODEL_PRESETS, TrainConfig, get_model_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Supervised fine-tuning for CyberSLM-2")
    p.add_argument("--preset", default="base-50m", choices=sorted(MODEL_PRESETS))
    p.add_argument("--pretrained", default="cyberslm2/checkpoints/pretrain/best.pt")
    p.add_argument("--train-data", default="cyberslm_sft/data/SFT.jsonl")
    p.add_argument("--val-data", default=None)
    p.add_argument("--tokenizer", default="tokenizer/tokenizer_output/tokenizer.model")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Muon LR; SFT wants roughly 10x lower than pretraining")
    p.add_argument("--adamw-lr", type=float, default=2e-5)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="adamw")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--no-train-on-reasoning", action="store_true",
                   help="mask <|think|> spans out of the loss")
    p.add_argument("--checkpoint-dir", default="cyberslm2/checkpoints/sft")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="only load the first N examples (debugging)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()

    mcfg = get_model_config(args.preset)
    mcfg.max_seq_len = args.seq_len
    # A little dropout is worth it here: SFT sets are small enough to memorize.
    mcfg.dropout = 0.05

    tcfg = TrainConfig(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
        lr=args.lr,
        adamw_lr=args.adamw_lr,
        warmup_steps=min(100, args.max_steps // 10),
        decay_frac=0.3,
        checkpoint_dir=args.checkpoint_dir,
        doc_aware_mask=False,        # SFT batches use padding masks instead
        eval_every=200,
        save_every=400,
    )

    print("=" * 70)
    print(f"CyberSLM-2 SFT plan  [{args.preset}]")
    print("=" * 70)
    print(f"  parameters     {mcfg.param_count()['total']:,}")
    print(f"  pretrained     {args.pretrained}")
    print(f"  train data     {args.train_data}")
    print(f"  tokenizer      {args.tokenizer}")
    print(f"  optimizer      {tcfg.optimizer} (lr={tcfg.lr}, adamw={tcfg.adamw_lr})")
    print(f"  steps          {tcfg.max_steps:,} "
          f"@ {tcfg.tokens_per_step():,} tokens/step")
    print(f"  train on <|think|>: {not args.no_train_on_reasoning}")
    print("=" * 70)

    if args.dry_run:
        print("\n--dry-run: nothing was allocated and no training was started.")
        return 0

    import sentencepiece as spm
    import torch
    from torch.utils.data import DataLoader, random_split

    from cyberslm2.data.datasets import SFTDataset, sft_collate
    from cyberslm2.data.special_tokens import PAD_ID, SpecialTokens
    from cyberslm2.model.transformer import CyberSLM2
    from cyberslm2.training.trainer import Trainer

    for path, what in ((args.tokenizer, "tokenizer"), (args.train_data, "train data")):
        if not Path(path).exists():
            print(f"ERROR: {what} not found at {path}")
            return 1

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    SpecialTokens.validate(sp)   # fails loudly on any id mismatch
    logging.info("tokenizer OK: vocab=%d", sp.get_piece_size())

    dataset = SFTDataset(
        args.train_data, sp, max_seq_len=args.seq_len,
        train_on_reasoning=not args.no_train_on_reasoning,
        limit=args.limit,
    )
    logging.info("loaded %d examples (%d skipped)", len(dataset), dataset.skipped)

    if args.val_data:
        train_ds = dataset
        val_ds = SFTDataset(args.val_data, sp, max_seq_len=args.seq_len)
    else:
        n_val = max(1, int(len(dataset) * args.val_frac))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(tcfg.seed),
        )
    logging.info("split: %d train / %d val", len(train_ds), len(val_ds))

    collate = lambda b: sft_collate(b, pad_id=PAD_ID)
    train_loader = DataLoader(train_ds, batch_size=tcfg.micro_batch_size,
                              shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.micro_batch_size,
                            shuffle=False, collate_fn=collate)

    model = CyberSLM2(mcfg)
    if Path(args.pretrained).exists():
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        # tolerate a torch.compile-wrapped checkpoint
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info("loaded pretrained weights (missing=%d unexpected=%d)",
                     len(missing), len(unexpected))
    else:
        logging.warning("No pretrained checkpoint at %s -- training from scratch, "
                        "which will not produce a usable instruct model.",
                        args.pretrained)

    print(model.summary())

    trainer = Trainer(model, mcfg, tcfg, train_loader, val_loader,
                      device=args.device, mode="sft")
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
