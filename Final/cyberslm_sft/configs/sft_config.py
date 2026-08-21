"""
CyberSLM SFT Configuration
===========================
Centralised dataclass for all hyperparameters and paths used
in the Supervised Instruction Fine-Tuning pipeline.

Path wiring
-----------
All default paths are resolved as absolute paths relative to the
``cyberslm_sft/`` project root so the pipeline works regardless of
where you invoke it from.

Quick setup — copy Stage 1 artifacts into the SFT project::

    # 1. Copy tokenizer
    cp "SLm Dataset/tokenizer/tokenizer_output/tokenizer.model" \\
       "SLm Dataset/cyberslm_sft/tokenizer/tokenizer.model"

    # 2. Copy pretrained weights
    cp "SLm Dataset/cyberslm/checkpoints/best.pt" \\
       "SLm Dataset/cyberslm_sft/checkpoints/pretrained/model.pt"

    # 3. Put your instruction dataset in data/
    #    data/train.jsonl  (required)
    #    data/val.jsonl    (optional)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# _PROJECT_ROOT = .../cyberslm_sft/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stage 1 dataset root (one level above cyberslm_sft/) — used only as fallback
_DATASET_ROOT = _PROJECT_ROOT.parent   # .../SLm Dataset/


# ---------------------------------------------------------------------------
# Model Architecture (must match pretrained checkpoint — never modify here)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Mirrors the pretrained CyberSLM architecture exactly.
    These values are frozen; do NOT change them for SFT.
    """
    vocab_size:   int   = 32_000
    hidden_size:  int   = 384
    num_layers:   int   = 12
    num_heads:    int   = 6
    head_dim:     int   = 64
    ffn_size:     int   = 1024
    max_seq_len:  int   = 2048   # MUST match the pretrained checkpoint
    weight_tying: bool  = True
    bias:         bool  = False
    dropout:      float = 0.0
    norm_eps:     float = 1e-6   # RMSNorm epsilon — must match pretraining


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    """
    Paths and special token ids for the SentencePiece BPE tokenizer.

    model_path
        Absolute path to ``tokenizer.model``.
        Default: ``<project_root>/tokenizer/tokenizer.model``
        (auto-falls back to Stage 1 canonical path if the local copy is absent)
    """
    model_path: str = str(_PROJECT_ROOT / "tokenizer" / "tokenizer.model")
    # Real SentencePiece token ids — verified against tokenizer.model
    pad_id: int = 0
    bos_id: int = 2   # <s>  (NOT 1)
    eos_id: int = 3   # </s> (NOT 2)
    unk_id: int = 1   # <unk>


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset paths and processing knobs."""
    train_path: str = str(_PROJECT_ROOT / "data" / "SFT.jsonl")
    val_path:   str = ""   # empty = auto-split from train_path using val_split

    # Maximum token length per sample (hard-truncate if exceeded).
    # Must be <= ModelConfig.max_seq_len.
    max_seq_len: int = 2048

    # Fraction of training data to use as validation when no val_path
    # is provided (ignored if val_path exists).
    val_split: float = 0.05

    # DataLoader workers (set 0 for debugging)
    num_workers:    int = 0
    prefetch_factor: Optional[int] = None

    # Whether to shuffle the training set each epoch
    shuffle: bool = True

    # Seed used for the val split and shuffling
    seed: int = 42

    # Cap on total samples loaded (-1 = no cap); useful for quick smoke tests
    max_samples: int = -1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Optimizer, scheduler, and loop settings.

    LR is intentionally much lower than typical pretraining (~1e-3 to 3e-4).
    SFT is refinement, not relearning.
    """
    # ---- Optimiser ----
    learning_rate: float = 2e-5          # Peak LR after warmup
    weight_decay:  float = 0.01
    beta1:         float = 0.9
    beta2:         float = 0.95
    eps:           float = 1e-8

    # ---- Gradient ----
    max_grad_norm:               float = 1.0
    gradient_accumulation_steps: int   = 4   # Effective batch = batch_size × accum

    # ---- Batch ----
    per_device_batch_size: int = 4

    # ---- Schedule ----
    num_epochs:    int   = 3
    warmup_ratio:  float = 0.03           # Fraction of total steps for warmup
    lr_schedule:   str   = "cosine"       # "cosine" | "linear" | "constant"
    min_lr_ratio:  float = 0.1            # min_lr = learning_rate × min_lr_ratio

    # ---- Precision ----
    dtype: str = "bfloat16"              # bf16 on GPU; auto-disabled on CPU

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Logging ----
    log_every_n_steps:  int = 10
    eval_every_n_steps: int = 200         # 0 = eval only at epoch end
    save_every_n_steps: int = 500         # 0 = save only at epoch end

    # ---- Output ----
    output_dir: str = str(_PROJECT_ROOT / "checkpoints")
    run_name:   str = "cyberslm-instruct"

    # ---- Resume ----
    resume_from_checkpoint: Optional[str] = None   # Path to checkpoint dir

    # ---- Base model ----
    pretrained_checkpoint: str = str(
        _PROJECT_ROOT / "checkpoints" / "pretrained" / "model.pt"
    )

    # ---- Inference sanity check ----
    run_inference_test: bool  = True
    max_new_tokens:     int   = 256
    temperature:        float = 0.7
    top_p:              float = 0.9


# ---------------------------------------------------------------------------
# Prompt / Template
# ---------------------------------------------------------------------------

@dataclass
class TemplateConfig:
    """
    Controls which prompt format is used to wrap instruction samples.
    The SFT loss is only computed on the *response* portion.
    """
    # Token that separates instruction/input from the response.
    # Loss masking begins immediately after this string (inclusive of the
    # newline that follows it).
    response_prefix: str = "### Response:\n"

    # NOTE: end-of-sequence is handled as a real token id (``<eos>`` == 3),
    # NOT a literal string. Appending the characters ``"</s>"`` would train
    # the model to emit text that the sampler never stops on. Kept as an empty
    # string so no literal EOS text is injected anywhere in the pipeline.
    eos_string: str = ""

    # Whether to strip leading/trailing whitespace from each field
    strip_fields: bool = True


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """
    Top-level configuration.  Pass this object through the entire pipeline.

    Usage::

        from configs.sft_config import SFTConfig, load_config, save_config

        cfg = SFTConfig()
        # or
        cfg = load_config("my_run/sft_config.json")
    """
    model:     ModelConfig     = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data:      DataConfig      = field(default_factory=DataConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    template:  TemplateConfig  = field(default_factory=TemplateConfig)

    def __post_init__(self) -> None:
        # Ensure max_seq_len for data never exceeds the model's context window
        if self.data.max_seq_len > self.model.max_seq_len:
            raise ValueError(
                f"DataConfig.max_seq_len ({self.data.max_seq_len}) exceeds "
                f"ModelConfig.max_seq_len ({self.model.max_seq_len}). "
                "Truncate to the model context window or below."
            )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_config(cfg: SFTConfig, path: str | Path) -> None:
    """Serialise an SFTConfig to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)


def load_config(path: str | Path) -> SFTConfig:
    """
    Deserialise an SFTConfig from a JSON file produced by ``save_config``.
    Provides forward-compatibility: unknown keys in the file are silently
    ignored so old checkpoints can still be loaded after field additions.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    def _filter(dc_type, d: dict) -> dict:
        """Return only keys that exist in the target dataclass."""
        valid = {f.name for f in dc_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return {k: v for k, v in d.items() if k in valid}

    model_cfg = ModelConfig(**_filter(ModelConfig, raw.get("model", {})))
    tok_cfg   = TokenizerConfig(**_filter(TokenizerConfig, raw.get("tokenizer", {})))
    data_cfg  = DataConfig(**_filter(DataConfig, raw.get("data", {})))
    train_cfg = TrainConfig(**_filter(TrainConfig, raw.get("train", {})))
    tmpl_cfg  = TemplateConfig(**_filter(TemplateConfig, raw.get("template", {})))

    return SFTConfig(
        model=model_cfg,
        tokenizer=tok_cfg,
        data=data_cfg,
        train=train_cfg,
        template=tmpl_cfg,
    )


def default_config() -> SFTConfig:
    """
    Return a default SFTConfig with all paths resolved to absolute paths
    inside the ``cyberslm_sft/`` project root.

    Tokenizer fallback
    ------------------
    If ``tokenizer/tokenizer.model`` doesn't exist locally, the function
    automatically falls back to the canonical Stage 1 path at
    ``../tokenizer/tokenizer_output/tokenizer.model``.
    """
    cfg = SFTConfig()

    # Auto-fallback to Stage 1 tokenizer if local copy is absent
    local_tok    = Path(cfg.tokenizer.model_path)
    canonical_tok = _DATASET_ROOT / "tokenizer" / "tokenizer_output" / "tokenizer.model"
    if not local_tok.exists() and canonical_tok.exists():
        import warnings
        warnings.warn(
            f"Local tokenizer not found at {local_tok}.\n"
            f"Falling back to Stage 1 tokenizer at {canonical_tok}.\n"
            "Copy it with:\n"
            f"  cp '{canonical_tok}' '{local_tok}'",
            stacklevel=2,
        )
        cfg.tokenizer.model_path = str(canonical_tok)

    return cfg
