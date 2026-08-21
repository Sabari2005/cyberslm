"""
presets.py
==========
Model and training configuration for CyberSLM-2.

Every field is validated at construction time (``__post_init__``) and the
parameter count is derived **analytically** in :meth:`ModelConfig.param_count`
so the architecture can be verified without allocating a single tensor. Run
``python -m cyberslm2.scripts.verify_architecture`` to check the arithmetic.

Design constraint: total parameters < 100,000,000.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def swiglu_hidden(d_model: int, mult: float = 8 / 3, multiple_of: int = 64) -> int:
    """
    Standard SwiGLU inner width.

    A SwiGLU FFN uses three matrices (gate, up, down) instead of the two a ReLU
    MLP uses, so the 4*d_model rule of thumb becomes (8/3)*d_model to keep the
    parameter count matched. Rounded up to a multiple of 64 to keep GEMM shapes
    tensor-core friendly.
    """
    h = int(mult * d_model)
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture hyperparameters for a CyberSLM-2 decoder-only transformer."""

    # --- core dimensions ---
    vocab_size: int = 32768          # power of 2: friendlier GEMM + room for special tokens
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12                # query heads
    n_kv_heads: int = 2              # GQA key/value heads (must divide n_heads)
    head_dim: int = 64
    ffn_hidden: int = 0              # 0 -> auto via swiglu_hidden()

    # --- context ---
    max_seq_len: int = 2048          # pretraining length
    rope_base: float = 10000.0
    rope_scaling: float = 1.0        # >1.0 stretches RoPE for long-context extension

    # --- normalization / numerics ---
    norm_eps: float = 1e-5
    qk_norm: bool = True             # RMSNorm on q and k before attention
    tie_embeddings: bool = True      # share input embedding with output head

    # --- regularization ---
    dropout: float = 0.0             # 0 is correct while data-bound by tokens, not epochs
    attn_dropout: float = 0.0

    # --- objective ---
    z_loss_coef: float = 1e-4        # stabilizes the softmax logit scale

    # --- init ---
    init_std: float = 0.02
    depth_scaled_init: bool = True   # scale residual-output projections by 1/sqrt(2*n_layers)

    def __post_init__(self) -> None:
        if self.ffn_hidden == 0:
            self.ffn_hidden = swiglu_hidden(self.d_model)

        # --- invariants that must hold for the maths to work ---
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads * head_dim must equal d_model: "
                f"{self.n_heads} * {self.head_dim} = {self.n_heads * self.head_dim} "
                f"!= {self.d_model}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for grouped-query attention"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.vocab_size <= 0 or self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, d_model and n_layers must all be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    # -- derived ------------------------------------------------------------

    @property
    def n_kv_groups(self) -> int:
        """How many query heads share each KV head."""
        return self.n_heads // self.n_kv_heads

    @property
    def kv_dim(self) -> int:
        """Width of the K and V projections (shrunk by GQA)."""
        return self.n_kv_heads * self.head_dim

    # -- analytic parameter accounting --------------------------------------

    def param_count(self) -> dict[str, int]:
        """
        Exact parameter count, derived from shapes rather than measured.

        All Linear layers are bias-free (standard for LLaMA-class models), so
        each contributes exactly in_features * out_features.
        """
        d, hd = self.d_model, self.head_dim
        kv = self.kv_dim

        embedding = self.vocab_size * d

        wq = d * (self.n_heads * hd)
        wk = d * kv
        wv = d * kv
        wo = (self.n_heads * hd) * d
        attn = wq + wk + wv + wo

        # optional RMSNorm on q/k: one gain vector per head_dim
        if self.qk_norm:
            attn += 2 * hd

        # SwiGLU: gate + up (d -> h) and down (h -> d)
        ffn = 3 * d * self.ffn_hidden

        # two pre-norms per block (attention + FFN), one gain vector each
        norms = 2 * d

        per_layer = attn + ffn + norms
        blocks = per_layer * self.n_layers
        final_norm = d

        # tied head costs nothing extra; untied adds a full V x d matrix
        lm_head = 0 if self.tie_embeddings else self.vocab_size * d

        total = embedding + blocks + final_norm + lm_head
        return {
            "embedding": embedding,
            "per_layer": per_layer,
            "blocks": blocks,
            "final_norm": final_norm,
            "lm_head": lm_head,
            "total": total,
            "non_embedding": total - embedding,
        }

    def kv_cache_bytes(self, seq_len: int, batch_size: int = 1, dtype_bytes: int = 2) -> int:
        """Bytes of KV cache at a given context length (bf16 = 2 bytes)."""
        return 2 * batch_size * self.n_layers * seq_len * self.kv_dim * dtype_bytes

    def flops_per_token(self) -> int:
        """
        Forward+backward FLOPs per training token, ~6*N_non_embedding plus
        attention's quadratic term. Used for compute budgeting.
        """
        n = self.param_count()["non_embedding"]
        dense = 6 * n
        attn_quadratic = 12 * self.n_layers * self.d_model * self.max_seq_len
        return dense + attn_quadratic

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Optimization and schedule hyperparameters."""

    # --- data ---
    train_bin: str = "tokenizer/data/train.bin"
    val_bin: str = "tokenizer/data/val.bin"
    seq_len: int = 2048

    # --- batching ---
    micro_batch_size: int = 8
    grad_accum_steps: int = 16       # effective tokens/step = mbs * accum * seq_len
    # 3200 * 262,144 = 839M tokens = ~4.1 passes over the 204.5M-token corpus.
    # Sized deliberately: past ~4 epochs a repeated token carries little new
    # signal, so extra steps buy overfitting rather than capability.
    max_steps: int = 3200

    # --- optimizer ---
    optimizer: Literal["muon", "adamw"] = "muon"
    lr: float = 3e-3                 # Muon LR for hidden matrices
    adamw_lr: float = 3e-4           # AdamW LR for embeddings / norms / scalars
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # --- schedule: warmup-stable-decay ---
    schedule: Literal["wsd", "cosine"] = "wsd"
    warmup_steps: int = 300          # ~9% of max_steps
    decay_frac: float = 0.2          # final 20% of steps decay to min_lr
    min_lr_frac: float = 0.0

    # --- precision ---
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    compile_model: bool = True

    # --- data-order / masking ---
    doc_aware_mask: bool = True      # block-diagonal attention within packed docs
    epochs_over_data: float = 4.0    # informational: repeats before returns decay

    # --- eval / io ---
    eval_every: int = 200
    eval_batches: int = 50
    save_every: int = 400
    checkpoint_dir: str = "cyberslm2/checkpoints"
    log_every: int = 10
    seed: int = 1337

    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    def total_tokens(self) -> int:
        return self.tokens_per_step() * self.max_steps

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _flagship() -> ModelConfig:
    """~98M params. Needs ~2B tokens to train properly (see README §Honesty)."""
    return ModelConfig(
        vocab_size=32768, d_model=768, n_layers=12,
        n_heads=12, n_kv_heads=2, head_dim=64,
        ffn_hidden=2048, max_seq_len=2048,
    )


def _data_matched() -> ModelConfig:
    """
    ~50M params. This is the *right* size for a 205M-token corpus at ~4 epochs
    (~16 tokens/param, close to compute-optimal). Recommended default.
    """
    return ModelConfig(
        vocab_size=32768, d_model=512, n_layers=12,
        n_heads=8, n_kv_heads=2, head_dim=64,
        ffn_hidden=1408, max_seq_len=2048,
    )


def _tiny() -> ModelConfig:
    """~5M params. CPU-only smoke testing on a laptop. Not a real model."""
    return ModelConfig(
        vocab_size=32768, d_model=128, n_layers=4,
        n_heads=4, n_kv_heads=1, head_dim=32,
        ffn_hidden=384, max_seq_len=256,
    )


MODEL_PRESETS = {
    "flagship-98m": _flagship,
    "base-50m": _data_matched,
    "tiny-5m": _tiny,
}

DEFAULT_PRESET = "base-50m"


def get_model_config(name: str = DEFAULT_PRESET) -> ModelConfig:
    if name not in MODEL_PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {sorted(MODEL_PRESETS)}"
        )
    return MODEL_PRESETS[name]()
