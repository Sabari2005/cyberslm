"""
CyberSLM Model Configuration
=============================
Defines the complete, validated configuration for the CyberSLM decoder-only
transformer. All hyperparameters are frozen at construction time and validated
for mathematical consistency before any model component is instantiated.

Architecture summary
--------------------
- Hidden dim      : 384
- Decoder layers  : 12
- Attention heads : 6
- Head dim        : 64  (hidden_dim / num_heads = 384 / 6 = 64)
- FFN inner dim   : 1024 (SwiGLU gate + value = 2 × 1024 → projects back to 384)
- Vocab size      : 32 000
- Max context     : 4 096
- Approx params   : 33.53 M
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CyberSLMConfig:
    """
    Immutable configuration for the CyberSLM decoder-only transformer.

    All fields are set once at construction; the frozen dataclass guarantees
    no accidental mutation during training. Call :meth:`validate` immediately
    after construction or use the convenience constructor
    :func:`default_config`.

    Attributes
    ----------
    vocab_size : int
        Number of tokens in the SentencePiece BPE vocabulary.
    max_seq_len : int
        Maximum token sequence length (context window).
    hidden_dim : int
        Embedding dimension ``d_model``.
    num_layers : int
        Number of stacked transformer decoder blocks.
    num_heads : int
        Number of attention heads.  Must evenly divide ``hidden_dim``.
    head_dim : int
        Dimension of each attention head.  Must equal ``hidden_dim // num_heads``.
    ffn_hidden_dim : int
        Inner dimension of the SwiGLU feed-forward network.
        The gate and value projections each map hidden_dim → ffn_hidden_dim,
        and the output projection maps ffn_hidden_dim → hidden_dim.
    rope_base : int
        Base for Rotary Position Embedding frequency computation (θ = 10 000).
    norm_eps : float
        Epsilon added inside RMSNorm to prevent division by zero.
    tie_weights : bool
        When True the output projection shares weights with the token embedding.
    bias : bool
        When True linear layers include a bias term (False = modern practice).
    dropout : float
        Residual / feed-forward dropout probability (0.0 = disabled).
    attn_dropout : float
        Attention weight dropout probability (0.0 = disabled).
    pad_token_id : Optional[int]
        Token ID used for padding; None if the dataset never pads.
    bos_token_id : Optional[int]
        Beginning-of-sequence token ID.
    eos_token_id : Optional[int]
        End-of-sequence token ID.
    """

    # ------------------------------------------------------------------ #
    # Vocabulary & sequence                                                #
    # ------------------------------------------------------------------ #
    vocab_size: int = 32_000
    max_seq_len: int = 4_096

    # ------------------------------------------------------------------ #
    # Transformer dimensions                                               #
    # ------------------------------------------------------------------ #
    hidden_dim: int = 384
    num_layers: int = 12
    num_heads: int = 6
    head_dim: int = 64        # must equal hidden_dim // num_heads
    ffn_hidden_dim: int = 1_024

    # ------------------------------------------------------------------ #
    # Positional encoding                                                  #
    # ------------------------------------------------------------------ #
    rope_base: int = 10_000

    # ------------------------------------------------------------------ #
    # Normalization                                                        #
    # ------------------------------------------------------------------ #
    norm_eps: float = 1e-6

    # ------------------------------------------------------------------ #
    # Architecture flags                                                   #
    # ------------------------------------------------------------------ #
    tie_weights: bool = True
    bias: bool = False
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Special token IDs (set by tokenizer integration layer)              #
    # ------------------------------------------------------------------ #
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = 2   # real SentencePiece BOS id
    eos_token_id: Optional[int] = 3   # real SentencePiece EOS id

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    def validate(self) -> "CyberSLMConfig":
        """
        Assert mathematical consistency of every hyperparameter.

        Raises
        ------
        ValueError
            If any hyperparameter violates an architectural constraint.

        Returns
        -------
        CyberSLMConfig
            Self, to allow chaining: ``cfg = CyberSLMConfig().validate()``.
        """
        errors: list[str] = []

        # Positivity checks
        for name, value in [
            ("vocab_size", self.vocab_size),
            ("max_seq_len", self.max_seq_len),
            ("hidden_dim", self.hidden_dim),
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
            ("head_dim", self.head_dim),
            ("ffn_hidden_dim", self.ffn_hidden_dim),
            ("rope_base", self.rope_base),
        ]:
            if value <= 0:
                errors.append(f"{name} must be positive, got {value}")

        # Attention head consistency
        if self.hidden_dim % self.num_heads != 0:
            errors.append(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        expected_head_dim = self.hidden_dim // self.num_heads
        if self.head_dim != expected_head_dim:
            errors.append(
                f"head_dim ({self.head_dim}) must equal "
                f"hidden_dim // num_heads = {expected_head_dim}"
            )

        # RoPE requires even head_dim (pairs of sin/cos)
        if self.head_dim % 2 != 0:
            errors.append(
                f"head_dim ({self.head_dim}) must be even for RoPE"
            )

        # Dropout bounds
        for name, value in [("dropout", self.dropout), ("attn_dropout", self.attn_dropout)]:
            if not (0.0 <= value < 1.0):
                errors.append(f"{name} must be in [0, 1), got {value}")

        # norm_eps positivity
        if self.norm_eps <= 0.0:
            errors.append(f"norm_eps must be positive, got {self.norm_eps}")

        if errors:
            raise ValueError(
                "CyberSLMConfig validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        return self

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #
    @property
    def total_attention_dim(self) -> int:
        """``num_heads × head_dim`` — equals ``hidden_dim`` by construction."""
        return self.num_heads * self.head_dim

    @property
    def rope_half_dim(self) -> int:
        """Number of frequency pairs in RoPE (``head_dim // 2``)."""
        return self.head_dim // 2

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        lines = [
            "CyberSLMConfig",
            "=" * 40,
            f"  vocab_size      : {self.vocab_size:,}",
            f"  max_seq_len     : {self.max_seq_len:,}",
            f"  hidden_dim      : {self.hidden_dim}",
            f"  num_layers      : {self.num_layers}",
            f"  num_heads       : {self.num_heads}",
            f"  head_dim        : {self.head_dim}",
            f"  ffn_hidden_dim  : {self.ffn_hidden_dim}",
            f"  rope_base       : {self.rope_base}",
            f"  norm_eps        : {self.norm_eps}",
            f"  tie_weights     : {self.tie_weights}",
            f"  bias            : {self.bias}",
            f"  dropout         : {self.dropout}",
            f"  attn_dropout    : {self.attn_dropout}",
        ]
        return "\n".join(lines)


def default_config() -> CyberSLMConfig:
    """
    Return the validated default CyberSLM configuration.

    This is the single source of truth for all training runs.
    All hyperparameters match the finalized architecture specification.

    Returns
    -------
    CyberSLMConfig
        A validated, immutable configuration object.
    """
    cfg = CyberSLMConfig()
    cfg.validate()
    return cfg
