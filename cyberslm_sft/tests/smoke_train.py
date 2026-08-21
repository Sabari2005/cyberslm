"""Tiny CPU SFT run to exercise the rewritten accumulation loop end to end."""
import sys, json, logging, shutil, torch
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "cyberslm_sft"))

from configs.sft_config import default_config
from data.prompt_formatter import PromptFormatter, Tokenizer
from data.sft_dataset import SFTDataset
from trainer import SFTTrainer
from model.cyberslm import CyberSLM as SFTModel

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    out = _ROOT / "_sft_smoke"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    try:
        cfg = default_config()
        cfg.tokenizer.model_path = str(_ROOT / "tokenizer" / "tokenizer_output" / "tokenizer.model")
        cfg.data.max_seq_len = 128
        cfg.data.num_workers = 0
        # tiny architecture so this is a unit test, not training
        cfg.model.vocab_size = 32000
        cfg.model.hidden_size = 64; cfg.model.num_layers = 2
        cfg.model.num_heads = 4;    cfg.model.head_dim = 16
        cfg.model.ffn_size = 128;   cfg.model.max_seq_len = 128
        cfg.train.output_dir = str(out)
        cfg.train.num_epochs = 1
        cfg.train.per_device_batch_size = 2
        cfg.train.gradient_accumulation_steps = 3
        cfg.train.eval_every_n_steps = 2
        cfg.train.save_every_n_steps = 0
        cfg.train.log_every_n_steps = 1
        cfg.train.run_inference_test = False
        cfg.train.dtype = "float32"

        tok = Tokenizer(cfg.tokenizer.model_path)
        fmt = PromptFormatter(cfg=cfg, tokenizer=tok)
        raw = [json.loads(l) for l in
               open(_ROOT / "cyberslm_sft" / "data" / "SFT.jsonl", encoding="utf-8").readlines()[:40]]
        train_ds = SFTDataset(raw[:32], fmt, "train")
        val_ds   = SFTDataset(raw[32:], fmt, "val")
        print(f"train={len(train_ds)} val={len(val_ds)}")

        model = SFTModel(cfg.model)
        t = SFTTrainer(model, train_ds, val_ds, cfg, torch.device("cpu"), tok)
        print(f"total_steps={t.total_steps} steps_per_epoch={t.steps_per_epoch}")
        t.train()
        print("\nRESULT")
        print("  global_step   :", t.state.global_step)
        print("  best_val_loss :", round(t.state.best_val_loss, 4))
        print("  best ckpt     :", (out / "best" / "model.pt").exists())
        print("  latest ckpt   :", (out / "latest" / "model.pt").exists())
    finally:
        shutil.rmtree(out, ignore_errors=True)

if __name__ == "__main__":
    main()
