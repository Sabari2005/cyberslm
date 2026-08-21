"""
Phase 1 self-contained tests.
Run with:  python -m pytest tests/test_phase1.py -v
or:        python tests/test_phase1.py
No GPU, no sentencepiece required.
"""

import json
import os
import sys
import tempfile
import unittest

# Make the project root importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.sft_config import (
    DataConfig,
    ModelConfig,
    SFTConfig,
    TemplateConfig,
    TrainConfig,
    default_config,
    load_config,
    save_config,
)
from data.dataset_loader import (
    _is_alpaca,
    _is_conversation,
    _normalise,
    _split_dataset,
    dataset_stats,
    load_jsonl,
)
from data.dataset_validator import (
    DatasetValidationError,
    validate_dataset,
)
from data.prompt_formatter import (
    IGNORE_INDEX,
    build_alpaca_segments,
    build_conversation_segments,
)


# ===========================================================================
# Config tests
# ===========================================================================

class TestSFTConfig(unittest.TestCase):

    def test_default_config_creates_successfully(self):
        cfg = default_config()
        self.assertIsInstance(cfg, SFTConfig)
        self.assertEqual(cfg.model.vocab_size, 32_000)
        self.assertEqual(cfg.model.hidden_size, 384)
        self.assertEqual(cfg.model.num_layers, 12)

    def test_max_seq_len_constraint(self):
        """DataConfig.max_seq_len must not exceed ModelConfig.max_seq_len."""
        with self.assertRaises(ValueError):
            SFTConfig(
                model=ModelConfig(max_seq_len=512),
                data=DataConfig(max_seq_len=1024),
            )

    def test_save_and_load_roundtrip(self):
        cfg = default_config()
        cfg.train.learning_rate = 3e-5
        cfg.train.num_epochs = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sft_config.json")
            save_config(cfg, path)
            loaded = load_config(path)

        self.assertAlmostEqual(loaded.train.learning_rate, 3e-5)
        self.assertEqual(loaded.train.num_epochs, 5)
        self.assertEqual(loaded.model.vocab_size, cfg.model.vocab_size)

    def test_load_config_ignores_unknown_keys(self):
        """Forward-compat: extra keys in JSON should not crash loading."""
        raw = {
            "model": {"vocab_size": 32000, "hidden_size": 384,
                       "num_layers": 12, "num_heads": 6, "head_dim": 64,
                       "ffn_size": 1024, "max_seq_len": 4096,
                       "weight_tying": True, "bias": False,
                       "dropout": 0.0, "norm_eps": 1e-5,
                       "FUTURE_FIELD": "ignored"},
            "tokenizer": {},
            "data": {},
            "train": {},
            "template": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cfg.json")
            with open(path, "w") as fh:
                json.dump(raw, fh)
            loaded = load_config(path)
        self.assertEqual(loaded.model.vocab_size, 32000)


# ===========================================================================
# Dataset loader tests
# ===========================================================================

class TestDatasetLoader(unittest.TestCase):

    def _write_jsonl(self, lines: list, path: str) -> None:
        with open(path, "w") as fh:
            for obj in lines:
                fh.write(json.dumps(obj) + "\n")

    def test_is_alpaca_positive(self):
        self.assertTrue(_is_alpaca({"instruction": "hi", "output": "hello"}))

    def test_is_alpaca_negative(self):
        self.assertFalse(_is_alpaca({"messages": []}))
        self.assertFalse(_is_alpaca({"instruction": "hi"}))  # missing output

    def test_is_conversation_positive(self):
        self.assertTrue(_is_conversation({"messages": [{"role": "user", "content": "hi"}]}))

    def test_normalise_alpaca(self):
        raw = {"instruction": "Explain X", "input": "context", "output": "answer"}
        result = _normalise(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["instruction"], "Explain X")
        self.assertEqual(result["output"], "answer")

    def test_normalise_alpaca_no_input(self):
        raw = {"instruction": "What is X?", "output": "X is ..."}
        result = _normalise(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["input"], "")

    def test_normalise_conversation(self):
        raw = {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]}
        result = _normalise(raw)
        self.assertIsNotNone(result)
        self.assertIn("messages", result)

    def test_normalise_unknown_returns_none(self):
        self.assertIsNone(_normalise({"foo": "bar"}))

    def test_load_jsonl_mixed_formats(self):
        samples = [
            {"instruction": "Q1", "output": "A1"},
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]},
            {"instruction": "Q2", "input": "ctx", "output": "A2"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(samples, path)
            loaded = load_jsonl(path)
        self.assertEqual(len(loaded), 3)

    def test_load_jsonl_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as fh:
                fh.write('{"instruction":"Q","output":"A"}\n')
                fh.write("\n")
                fh.write("# comment\n")
                fh.write('{"instruction":"Q2","output":"A2"}\n')
            loaded = load_jsonl(path)
        self.assertEqual(len(loaded), 2)

    def test_load_jsonl_skips_malformed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as fh:
                fh.write('{"instruction":"Q","output":"A"}\n')
                fh.write("NOT JSON\n")
                fh.write('{"instruction":"Q2","output":"A2"}\n')
            loaded = load_jsonl(path)
        # malformed line skipped → 2 valid samples
        self.assertEqual(len(loaded), 2)

    def test_load_jsonl_max_samples(self):
        samples = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(20)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(samples, path)
            loaded = load_jsonl(path, max_samples=5)
        self.assertEqual(len(loaded), 5)

    def test_split_dataset_sizes(self):
        samples = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(100)]
        train, val = _split_dataset(samples, val_ratio=0.1, seed=42)
        self.assertEqual(len(train) + len(val), 100)
        self.assertEqual(len(val), 10)

    def test_split_dataset_deterministic(self):
        samples = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(50)]
        _, val1 = _split_dataset(samples, val_ratio=0.1, seed=99)
        _, val2 = _split_dataset(samples, val_ratio=0.1, seed=99)
        self.assertEqual(val1, val2)

    def test_split_dataset_different_seeds(self):
        samples = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(50)]
        _, val1 = _split_dataset(samples, val_ratio=0.2, seed=1)
        _, val2 = _split_dataset(samples, val_ratio=0.2, seed=2)
        # Different seeds should (almost certainly) produce different splits
        self.assertNotEqual(val1, val2)

    def test_dataset_stats(self):
        samples = [
            {"instruction": "Q", "output": "A"},
            {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
        ]
        stats = dataset_stats(samples)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["alpaca_format"], 1)
        self.assertEqual(stats["conv_format"], 1)


# ===========================================================================
# Dataset validator tests
# ===========================================================================

class TestDatasetValidator(unittest.TestCase):

    def _alpaca(self, instruction="Q", output="A", input_text=""):
        return {"instruction": instruction, "input": input_text, "output": output}

    def _conv(self, messages=None):
        if messages is None:
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        return {"messages": messages}

    def test_valid_alpaca_passes(self):
        samples = [self._alpaca("What is SQL injection?", "It is a ...", "")]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 1)
        self.assertEqual(report.invalid, 0)

    def test_valid_conversation_passes(self):
        samples = [self._conv()]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 1)
        self.assertEqual(report.invalid, 0)

    def test_empty_instruction_is_invalid(self):
        samples = [self._alpaca(instruction="", output="A")]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 0)
        self.assertEqual(report.invalid, 1)

    def test_empty_output_is_invalid(self):
        samples = [self._alpaca(instruction="Q", output="")]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 0)
        self.assertEqual(report.invalid, 1)

    def test_whitespace_only_instruction_is_invalid(self):
        samples = [self._alpaca(instruction="   ", output="A")]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 0)

    def test_conversation_no_assistant_is_invalid(self):
        samples = [self._conv(messages=[
            {"role": "user", "content": "Hello"},
        ])]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 0)
        self.assertEqual(report.invalid, 1)

    def test_conversation_no_user_is_invalid(self):
        samples = [self._conv(messages=[
            {"role": "assistant", "content": "Hello"},
        ])]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 0)

    def test_conversation_empty_messages_is_invalid(self):
        samples = [{"messages": []}]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 0)

    def test_conversation_bad_role_is_invalid(self):
        samples = [self._conv(messages=[
            {"role": "narrator", "content": "Once upon a time"},
            {"role": "assistant", "content": "A"},
        ])]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 0)

    def test_strict_mode_raises(self):
        samples = [self._alpaca(instruction="", output="A")]
        with self.assertRaises(DatasetValidationError):
            validate_dataset(samples, strict=True)

    def test_mixed_valid_invalid(self):
        samples = [
            self._alpaca("Good Q", "Good A"),
            self._alpaca("", "A"),      # invalid
            self._conv(),               # valid
            {"messages": []},           # invalid
        ]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 2)
        self.assertEqual(report.invalid, 2)

    def test_input_field_not_required(self):
        """'input' is optional for alpaca samples."""
        sample = {"instruction": "What is X?", "output": "X is Y."}
        valid, report = validate_dataset([sample])
        self.assertEqual(len(valid), 1)

    def test_system_role_in_conversation_allowed(self):
        samples = [self._conv(messages=[
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ])]
        valid, report = validate_dataset(samples)
        self.assertEqual(len(valid), 1)


# ===========================================================================
# Prompt formatter tests (no tokenizer needed)
# ===========================================================================

class TestPromptFormatter(unittest.TestCase):
    """
    Tests for the segment-based formatter. Special tokens (BOS/EOS) are added
    as token *ids* downstream by ``encode_segments``, so the rendered segment
    text never contains literal ``<s>``/``</s>`` strings.
    """

    def setUp(self):
        self.tmpl = TemplateConfig()

    # ---- Alpaca ----

    def test_alpaca_with_input_contains_section_headers(self):
        segs = build_alpaca_segments(
            instruction="Summarise this.",
            output="Summary here.",
            input_text="Long text goes here.",
            tmpl=self.tmpl,
        )
        full = "".join(t for t, _ in segs)
        self.assertIn("### Instruction:", full)
        self.assertIn("### Input:", full)
        self.assertIn("### Response:", full)
        self.assertIn("Summary here.", full)
        # No literal EOS string is injected — EOS is a token id added later.
        self.assertNotIn("</s>", full)

    def test_alpaca_without_input_omits_input_section(self):
        segs = build_alpaca_segments(
            instruction="What is X?", output="X is Y.", input_text="", tmpl=self.tmpl
        )
        full = "".join(t for t, _ in segs)
        self.assertNotIn("### Input:", full)
        self.assertIn("### Instruction:", full)
        self.assertIn("### Response:", full)

    def test_alpaca_prompt_only_excludes_response(self):
        segs = build_alpaca_segments(
            instruction="Q", output="A", tmpl=self.tmpl, include_response=False
        )
        # No loss-target (response) segment should be present.
        self.assertFalse(any(is_target for _, is_target in segs))
        full = "".join(t for t, _ in segs)
        self.assertIn("### Response:", full)

    def test_alpaca_response_is_single_loss_target(self):
        output = "This is the answer."
        segs = build_alpaca_segments(instruction="Q", output=output, tmpl=self.tmpl)
        targets = [t for t, is_target in segs if is_target]
        self.assertEqual(targets, [output])

    def test_alpaca_strip_fields(self):
        segs = build_alpaca_segments(
            instruction="  Q  ", output="  A  ", tmpl=TemplateConfig(strip_fields=True)
        )
        full = "".join(t for t, _ in segs)
        self.assertIn("### Instruction:\nQ\n", full)

    # ---- Conversation ----

    def test_conversation_basic_structure(self):
        messages = [
            {"role": "user",      "content": "What is SQL injection?"},
            {"role": "assistant", "content": "SQL injection is ..."},
        ]
        segs = build_conversation_segments(messages, tmpl=self.tmpl)
        full = "".join(t for t, _ in segs)
        self.assertIn("### User:", full)
        self.assertIn("### Assistant:", full)
        self.assertIn("SQL injection is ...", full)
        self.assertTrue(any(is_target for _, is_target in segs))

    def test_conversation_system_turn_included(self):
        messages = [
            {"role": "system",    "content": "You are a cybersecurity expert."},
            {"role": "user",      "content": "Explain AES."},
            {"role": "assistant", "content": "AES is ..."},
        ]
        segs = build_conversation_segments(messages, tmpl=self.tmpl)
        full = "".join(t for t, _ in segs)
        self.assertIn("System:", full)
        self.assertIn("AES is ...", full)

    def test_conversation_multiturn_targets(self):
        messages = [
            {"role": "user",      "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user",      "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        segs = build_conversation_segments(messages, tmpl=self.tmpl)
        targets = [t for t, is_target in segs if is_target]
        self.assertEqual(targets, ["A1", "A2"])

    def test_conversation_inference_mode_no_final_response(self):
        messages = [
            {"role": "user",      "content": "What is X?"},
            {"role": "assistant", "content": "X is ..."},
        ]
        segs = build_conversation_segments(
            messages, tmpl=self.tmpl, include_final_response=False
        )
        full = "".join(t for t, _ in segs)
        self.assertNotIn("X is ...", full)     # response body omitted
        self.assertIn("### Assistant:", full)  # prefix present

    # ---- IGNORE_INDEX constant ----

    def test_ignore_index_value(self):
        """nn.CrossEntropyLoss default ignore_index is -100."""
        self.assertEqual(IGNORE_INDEX, -100)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [TestSFTConfig, TestDatasetLoader, TestDatasetValidator, TestPromptFormatter]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
