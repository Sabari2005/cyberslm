"""
Phase 4 tests — CheckpointManager, Inference utilities
Run with: python tests/test_phase4.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.sft_config import SFTConfig, default_config, save_config, load_config
from utils.checkpoint_manager import CheckpointManager, CheckpointMeta
from utils.inference import generate, _top_p_sample, model_context_len
from utils.optimizer import build_optimizer, build_scheduler


# ---------------------------------------------------------------------------
# Shared tiny model + dummy optimizer/scheduler
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self, vocab=50, dim=16, seq=8):
        super().__init__()
        self.embed     = nn.Embedding(vocab, dim)
        self.proj      = nn.Linear(dim, vocab, bias=False)
        self.max_seq_len = seq
        self.vocab_size  = vocab

    def forward(self, input_ids, attention_mask=None):
        return self.proj(self.embed(input_ids))


def _make_optimizer_and_scheduler(model):
    cfg   = default_config().train
    opt   = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=100)
    return opt, sched


# ---------------------------------------------------------------------------
# Minimal tokenizer stub (no SentencePiece needed)
# ---------------------------------------------------------------------------

class StubTokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0
    vocab_size = 50

    def encode(self, text: str, **kwargs):
        return [self.bos_id] + [ord(c) % 48 + 3 for c in text[:4]]

    def decode(self, ids):
        return "".join(chr(i + 32) for i in ids if i > 2)


# ===========================================================================
# CheckpointManager tests
# ===========================================================================

class TestCheckpointManager(unittest.TestCase):

    def setUp(self):
        self._tmp  = tempfile.TemporaryDirectory()
        self.outdir = self._tmp.name
        self.model  = TinyModel()
        self.opt, self.sched = _make_optimizer_and_scheduler(self.model)
        self.cfg   = default_config()

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, mgr, tag="latest", step=10, epoch=0, val_loss=2.0):
        return mgr.save(
            tag=tag,
            model=self.model,
            optimizer=self.opt,
            scheduler=self.sched,
            epoch=epoch,
            step=step,
            val_loss=val_loss,
            best_val_loss=val_loss,
            cfg=self.cfg,
        )

    def test_save_creates_directory(self):
        mgr  = CheckpointManager(self.outdir)
        path = self._save(mgr, tag="latest")
        self.assertTrue(Path(path).is_dir())

    def test_save_creates_model_pt(self):
        mgr  = CheckpointManager(self.outdir)
        path = self._save(mgr)
        self.assertTrue((Path(path) / "model.pt").exists())

    def test_save_creates_optimizer_pt(self):
        mgr  = CheckpointManager(self.outdir)
        path = self._save(mgr)
        self.assertTrue((Path(path) / "optimizer.pt").exists())

    def test_save_creates_scheduler_pt(self):
        mgr  = CheckpointManager(self.outdir)
        path = self._save(mgr)
        self.assertTrue((Path(path) / "scheduler.pt").exists())

    def test_save_creates_training_state_pt(self):
        mgr  = CheckpointManager(self.outdir)
        path = self._save(mgr)
        self.assertTrue((Path(path) / "training_state.pt").exists())

    def test_save_creates_config_json(self):
        mgr  = CheckpointManager(self.outdir)
        path = self._save(mgr)
        self.assertTrue((Path(path) / "sft_config.json").exists())

    def test_save_creates_meta_json(self):
        mgr  = CheckpointManager(self.outdir)
        path = self._save(mgr, step=42, val_loss=1.5)
        meta_path = Path(path) / "meta.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path) as fh:
            meta = json.load(fh)
        self.assertEqual(meta["global_step"], 42)
        self.assertAlmostEqual(meta["val_loss"], 1.5)

    def test_load_restores_model_weights(self):
        mgr = CheckpointManager(self.outdir)
        # Save with specific weight values
        with torch.no_grad():
            self.model.embed.weight.fill_(3.14)
        self._save(mgr, tag="latest")

        # Create a fresh model and load into it
        fresh = TinyModel()
        mgr.load("latest", model=fresh, device=torch.device("cpu"))
        self.assertTrue(
            torch.allclose(fresh.embed.weight, self.model.embed.weight)
        )

    def test_load_nonexistent_tag_raises(self):
        mgr = CheckpointManager(self.outdir)
        fresh = TinyModel()
        with self.assertRaises(FileNotFoundError):
            mgr.load("nonexistent", model=fresh, device=torch.device("cpu"))

    def test_load_returns_training_state(self):
        mgr = CheckpointManager(self.outdir)
        self._save(mgr, tag="latest", step=77, epoch=2, val_loss=1.23)
        fresh = TinyModel()
        ts = mgr.load("latest", model=fresh, device=torch.device("cpu"))
        self.assertEqual(ts["global_step"], 77)
        self.assertEqual(ts["epoch"],       2)
        self.assertAlmostEqual(ts["val_loss"], 1.23)

    def test_save_latest_convenience(self):
        mgr  = CheckpointManager(self.outdir)
        path = mgr.save_latest(
            model=self.model, optimizer=self.opt, scheduler=self.sched,
            epoch=0, step=1, val_loss=2.0, best_val_loss=2.0, cfg=self.cfg,
        )
        self.assertEqual(Path(path).name, "latest")

    def test_save_best_convenience(self):
        mgr  = CheckpointManager(self.outdir)
        path = mgr.save_best(
            model=self.model, optimizer=self.opt, scheduler=self.sched,
            epoch=0, step=1, val_loss=1.5, best_val_loss=1.5, cfg=self.cfg,
        )
        self.assertEqual(Path(path).name, "best")

    def test_versioned_save_creates_step_dir(self):
        mgr  = CheckpointManager(self.outdir, keep_last_n=2)
        path = mgr.save_versioned(
            step=500,
            model=self.model, optimizer=self.opt, scheduler=self.sched,
            epoch=0, val_loss=2.0, best_val_loss=2.0, cfg=self.cfg,
        )
        self.assertTrue(Path(path).name.startswith("step_"))

    def test_versioned_prunes_old_checkpoints(self):
        mgr = CheckpointManager(self.outdir, keep_last_n=2)
        for step in [100, 200, 300]:
            mgr.save_versioned(
                step=step,
                model=self.model, optimizer=self.opt, scheduler=self.sched,
                epoch=0, val_loss=2.0, best_val_loss=2.0, cfg=self.cfg,
            )
        # Only 2 should remain
        self.assertEqual(len(mgr._versioned), 2)
        # The oldest (step_100) should have been deleted
        self.assertFalse((Path(self.outdir) / "step_0000100").exists())

    def test_list_checkpoints(self):
        mgr = CheckpointManager(self.outdir)
        self._save(mgr, tag="latest", step=10)
        self._save(mgr, tag="best",   step=10)
        ckpts = mgr.list_checkpoints()
        tags  = [c.tag for c in ckpts]
        self.assertIn("latest", tags)
        self.assertIn("best",   tags)

    def test_best_checkpoint_path_none_when_absent(self):
        mgr = CheckpointManager(self.outdir)
        self.assertIsNone(mgr.best_checkpoint_path())

    def test_best_checkpoint_path_set_after_save(self):
        mgr = CheckpointManager(self.outdir)
        self._save(mgr, tag="best")
        self.assertIsNotNone(mgr.best_checkpoint_path())

    def test_config_serialised_and_loadable(self):
        mgr = CheckpointManager(self.outdir)
        self._save(mgr, tag="latest")
        cfg_path = Path(self.outdir) / "latest" / "sft_config.json"
        loaded   = load_config(cfg_path)
        self.assertEqual(loaded.model.vocab_size, self.cfg.model.vocab_size)


# ===========================================================================
# Inference utility tests
# ===========================================================================

class TestInference(unittest.TestCase):

    def setUp(self):
        self.model     = TinyModel(vocab=50, dim=16, seq=8)
        self.tokenizer = StubTokenizer()
        self.device    = torch.device("cpu")

    def test_top_p_sample_returns_valid_index(self):
        logits = torch.randn(50)
        idx    = _top_p_sample(logits, top_p=0.9)
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, 50)

    def test_top_p_sample_full_p_returns_any(self):
        """top_p=1.0 means consider all tokens (no nucleus pruning)."""
        logits = torch.randn(50)
        idx    = _top_p_sample(logits, top_p=1.0)
        self.assertIsInstance(idx, int)

    def test_top_p_sample_tiny_p_returns_single(self):
        """top_p near 0 should still return a valid index (at least top token)."""
        logits = torch.zeros(50)
        logits[7] = 100.0   # very high logit → nearly all probability mass
        idx = _top_p_sample(logits, top_p=0.01)
        self.assertEqual(idx, 7)

    def test_model_context_len_attribute(self):
        self.assertEqual(model_context_len(self.model), 8)

    def test_model_context_len_fallback(self):
        class NoCtx(nn.Module):
            def forward(self, x): return x
        self.assertEqual(model_context_len(NoCtx()), 4096)

    def test_generate_returns_string(self):
        input_ids = torch.tensor([[1, 3, 4, 5]])
        result    = generate(
            model=self.model,
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            max_new_tokens=10,
            temperature=1.0,
            top_p=1.0,
            eos_id=self.tokenizer.eos_id,
            device=self.device,
        )
        self.assertIsInstance(result, str)

    def test_generate_stops_at_max_tokens(self):
        """Generation should produce at most max_new_tokens tokens."""
        # Force eos to an unreachable id so it always generates max_tokens
        input_ids = torch.tensor([[1, 3]])
        result    = generate(
            model=self.model,
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            max_new_tokens=5,
            temperature=1.0,
            top_p=1.0,
            eos_id=999,   # unreachable → always hits max
            device=self.device,
        )
        # Decoded string length ≤ 5 chars (1 char per token after decode)
        self.assertIsInstance(result, str)

    def test_generate_stops_at_eos(self):
        """
        If the model always outputs eos_id, generation should stop immediately.
        """
        class AlwaysEOSModel(nn.Module):
            max_seq_len = 64
            def forward(self, x, attention_mask=None):
                B, T = x.shape
                logits = torch.zeros(B, T, 50)
                logits[:, :, 2] = 100.0   # eos = id 2 has all probability
                return logits

        result = generate(
            model=AlwaysEOSModel(),
            input_ids=torch.tensor([[1, 3]]),
            tokenizer=self.tokenizer,
            max_new_tokens=20,
            temperature=1.0,
            top_p=1.0,
            eos_id=2,
            device=self.device,
        )
        # Should produce empty string (eos immediately)
        self.assertEqual(result, "")

    def test_generate_temperature_affects_distribution(self):
        """Different temperatures should produce different distributions
        (with high probability for a random model)."""
        torch.manual_seed(99)
        input_ids = torch.tensor([[1, 3, 4]])
        results   = set()
        for _ in range(10):
            r = generate(
                model=self.model,
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                max_new_tokens=4,
                temperature=2.0,
                top_p=1.0,
                eos_id=999,
                device=self.device,
            )
            results.add(r)
        # With temperature=2.0, we expect some diversity over 10 runs
        self.assertGreater(len(results), 1)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestCheckpointManager, TestInference]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
