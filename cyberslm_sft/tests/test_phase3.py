"""
Phase 3 tests — Optimizer, Scheduler, Validation loop
Run with: python tests/test_phase3.py
"""

from __future__ import annotations

import os
import sys
import math
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.sft_config import TrainConfig, SFTConfig
from utils.optimizer import (
    build_optimizer,
    build_scheduler,
    clip_gradients,
    _get_param_groups,
)
from utils.validation import run_validation, ValidationResult
from utils.seed import set_seed
from data.loss_masking import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Tiny model for testing
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """A minimal model: Embedding → Linear → logits. Used for all Phase 3 tests."""
    def __init__(self, vocab=50, dim=16, seq=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj  = nn.Linear(dim, vocab, bias=False)
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        return self.proj(self.embed(input_ids))   # (B, T, V)


# ---------------------------------------------------------------------------
# Fake DataLoader for validation tests
# ---------------------------------------------------------------------------

def _make_val_loader(n_batches=4, B=2, T=8, V=50):
    """Returns a list of batch dicts simulating a DataLoader."""
    batches = []
    for _ in range(n_batches):
        input_ids = torch.randint(0, V, (B, T))
        labels    = input_ids.clone()
        # Mask first 3 positions
        labels[:, :3] = IGNORE_INDEX
        batches.append({
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        })
    return batches


# ===========================================================================
# Optimizer tests
# ===========================================================================

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.model = TinyModel()
        self.cfg = TrainConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

    def test_build_optimizer_returns_adamw(self):
        opt = build_optimizer(self.model, self.cfg)
        self.assertIsInstance(opt, torch.optim.AdamW)

    def test_optimizer_has_two_param_groups(self):
        opt = build_optimizer(self.model, self.cfg)
        self.assertEqual(len(opt.param_groups), 2)

    def test_decay_group_has_weight_decay(self):
        opt = build_optimizer(self.model, self.cfg)
        decay_group = opt.param_groups[0]
        self.assertAlmostEqual(decay_group["weight_decay"], 0.01)

    def test_no_decay_group_has_zero_weight_decay(self):
        opt = build_optimizer(self.model, self.cfg)
        no_decay_group = opt.param_groups[1]
        self.assertAlmostEqual(no_decay_group["weight_decay"], 0.0)

    def test_optimizer_lr_set_correctly(self):
        opt = build_optimizer(self.model, self.cfg)
        for pg in opt.param_groups:
            self.assertAlmostEqual(pg["lr"], 2e-5)

    def test_param_groups_cover_all_parameters(self):
        groups = _get_param_groups(self.model, weight_decay=0.01)
        total_from_groups = sum(p.numel() for g in groups for p in g["params"])
        total_model = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_from_groups, total_model)

    def test_no_duplicate_parameters_in_groups(self):
        groups = _get_param_groups(self.model, weight_decay=0.01)
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate parameters found in groups")

    def test_clip_gradients_returns_norm(self):
        model = TinyModel()
        x = torch.randint(0, 50, (1, 8))
        loss = model(x).sum()
        loss.backward()
        norm = clip_gradients(model, max_grad_norm=1.0)
        self.assertIsInstance(norm, float)
        self.assertGreaterEqual(norm, 0.0)

    def test_clip_gradients_clips_large_grads(self):
        model = TinyModel()
        # Manually set large gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 100.0
        clip_gradients(model, max_grad_norm=1.0)
        # After clipping, all grad norms should be ≤ 1.0
        total_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5
        self.assertLessEqual(total_norm, 1.01)  # small tolerance


# ===========================================================================
# Scheduler tests
# ===========================================================================

class TestScheduler(unittest.TestCase):

    def _build(self, schedule="cosine", total=100, warmup_ratio=0.1, min_lr_ratio=0.1):
        model = TinyModel()
        cfg   = TrainConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
            lr_schedule=schedule,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        opt   = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg, total_steps=total)
        return opt, sched

    def _collect_lrs(self, opt, sched, total):
        lrs = []
        for _ in range(total):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        return lrs

    def test_cosine_warmup_increases(self):
        opt, sched = self._build("cosine", total=100, warmup_ratio=0.1)
        lrs = self._collect_lrs(opt, sched, 10)
        # LR should monotonically increase during warmup
        for i in range(1, len(lrs)):
            self.assertGreaterEqual(lrs[i], lrs[i - 1])

    def test_cosine_decays_after_warmup(self):
        opt, sched = self._build("cosine", total=100, warmup_ratio=0.1)
        lrs = self._collect_lrs(opt, sched, 100)
        # Peak LR should be around step 10
        # LR at end should be lower than peak
        self.assertLess(lrs[-1], lrs[10])

    def test_cosine_min_lr_floor(self):
        opt, sched = self._build("cosine", total=200, warmup_ratio=0.05, min_lr_ratio=0.1)
        lrs = self._collect_lrs(opt, sched, 200)
        base_lr = 1e-3
        min_lr  = base_lr * 0.1
        # Skip step 0 (before first optimizer.step(): LambdaLR initial call)
        # All post-warmup LRs should be ≥ min_lr
        warmup_end = int(200 * 0.05) + 1
        for lr in lrs[warmup_end:]:
            self.assertGreaterEqual(lr, min_lr - 1e-10)

    def test_linear_schedule_decays(self):
        opt, sched = self._build("linear", total=100, warmup_ratio=0.1)
        lrs = self._collect_lrs(opt, sched, 100)
        self.assertLess(lrs[-1], lrs[10])

    def test_constant_schedule_flat_after_warmup(self):
        opt, sched = self._build("constant", total=100, warmup_ratio=0.1)
        lrs = self._collect_lrs(opt, sched, 100)
        # After warmup, all LRs should be equal
        post_warmup = lrs[11:]
        self.assertAlmostEqual(max(post_warmup) - min(post_warmup), 0.0, places=8)

    def test_invalid_schedule_raises(self):
        model = TinyModel()
        cfg   = TrainConfig(lr_schedule="invalid_schedule")
        opt   = build_optimizer(model, cfg)
        with self.assertRaises(ValueError):
            build_scheduler(opt, cfg, total_steps=100)

    def test_scheduler_step_count(self):
        opt, sched = self._build("cosine", total=50)
        lrs_before = sched.get_last_lr()[0]
        sched.step()
        lrs_after = sched.get_last_lr()[0]
        # After one step (still in warmup for ratio=0.1, total=50),
        # LR should have changed
        self.assertNotEqual(lrs_before, lrs_after)


# ===========================================================================
# Validation loop tests
# ===========================================================================

class TestValidation(unittest.TestCase):

    def setUp(self):
        set_seed(42)
        self.model  = TinyModel(vocab=50, dim=16, seq=8)
        self.device = torch.device("cpu")

    def test_run_validation_returns_result(self):
        loader = _make_val_loader(n_batches=3)
        result = run_validation(self.model, loader, self.device)
        self.assertIsInstance(result, ValidationResult)

    def test_val_loss_is_positive(self):
        loader = _make_val_loader(n_batches=4)
        result = run_validation(self.model, loader, self.device)
        self.assertGreater(result.val_loss, 0.0)

    def test_val_ppl_is_exp_of_loss(self):
        loader = _make_val_loader(n_batches=4)
        result = run_validation(self.model, loader, self.device)
        expected_ppl = math.exp(min(result.val_loss, 20.0))
        self.assertAlmostEqual(result.val_ppl, expected_ppl, places=4)

    def test_val_ppl_greater_than_one(self):
        loader = _make_val_loader(n_batches=2)
        result = run_validation(self.model, loader, self.device)
        self.assertGreater(result.val_ppl, 1.0)

    def test_tok_per_sec_positive(self):
        loader = _make_val_loader(n_batches=4)
        result = run_validation(self.model, loader, self.device)
        self.assertGreater(result.tok_per_sec, 0.0)

    def test_n_batches_correct(self):
        loader = _make_val_loader(n_batches=5)
        result = run_validation(self.model, loader, self.device)
        self.assertEqual(result.n_batches, 5)

    def test_max_batches_limits_evaluation(self):
        loader = _make_val_loader(n_batches=10)
        result = run_validation(self.model, loader, self.device, max_batches=3)
        self.assertEqual(result.n_batches, 3)

    def test_model_back_to_train_mode_after_validation(self):
        self.model.train()
        loader = _make_val_loader(n_batches=2)
        run_validation(self.model, loader, self.device)
        self.assertTrue(self.model.training)

    def test_fully_masked_batches_skipped(self):
        """Batches with all -100 labels should be skipped without error."""
        batch = {
            "input_ids":      torch.randint(0, 50, (2, 8)),
            "labels":         torch.full((2, 8), IGNORE_INDEX, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        result = run_validation(self.model, [batch], self.device)
        # Should produce inf loss (0 active tokens) without crashing
        self.assertEqual(result.tokens_seen, 0)

    def test_validation_is_deterministic(self):
        """Same model + same data → same loss."""
        loader = _make_val_loader(n_batches=3)
        set_seed(0)
        r1 = run_validation(self.model, loader, self.device)
        set_seed(0)
        r2 = run_validation(self.model, loader, self.device)
        self.assertAlmostEqual(r1.val_loss, r2.val_loss, places=5)

    def test_tokens_seen_counts_active_tokens(self):
        B, T = 2, 8
        mask_count = 3
        active_per_batch = B * (T - mask_count)
        n_batches = 4
        loader = _make_val_loader(n_batches=n_batches, B=B, T=T)
        result = run_validation(self.model, loader, self.device)
        self.assertEqual(result.tokens_seen, active_per_batch * n_batches)


# ===========================================================================
# Seed tests
# ===========================================================================

class TestSeed(unittest.TestCase):

    def test_set_seed_deterministic_torch(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        self.assertTrue(torch.allclose(a, b))

    def test_different_seeds_produce_different_tensors(self):
        set_seed(1)
        a = torch.randn(5)
        set_seed(2)
        b = torch.randn(5)
        self.assertFalse(torch.allclose(a, b))


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    loader_ = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [TestOptimizer, TestScheduler, TestValidation, TestSeed]:
        suite.addTests(loader_.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
