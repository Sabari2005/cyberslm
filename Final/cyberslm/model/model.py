"""
CyberSLM — Complete Decoder-Only Language Model
================================================
Assembles all components into the full model:

  Token Embedding  →  Decoder Stack (×12)  →  Final RMSNorm  →  LM Head

Weight tying
------------
The LM head (output projection that maps hidden_dim → vocab_size) shares its
weight matrix with the token embedding (vocab_size × hidden_dim).

Mathematical justification: both the embedding matrix E and the unembedding
matrix U operate in the same semantic space. Tying U = Eᵀ forces consistency
("a token's output representation should be similar to its input representation"),
reduces parameters by vocab_size × hidden_dim = 32 000 × 384 ≈ 12.3 M, and
empirically improves perplexity on small models.

Parameter count breakdown
--------------------------
Component                                Params
---------------------------------------- ------
Token embedding (vocab × hidden)          12 288 000
  ↳ shared with LM head (no extra cost)          0
Decoder blocks × 12:
  attn_norm (RMSNorm)          384 ×12 =    4 608
  attn Q/K/V/O proj       589 824 ×12 =  7 077 888
  ffn_norm (RMSNorm)           384 ×12 =    4 608
  ffn gate/val/out        1 179 648 ×12 = 14 155 776
Final RMSNorm                               384
---------------------------------------- ------
Total                                  ≈ 33 531 264  (≈33.53 M)

Note: RoPE buffers and causal mask buffers are NOT parameters.

Initialisation
--------------
- Embeddings: N(0, 0.02)  — small but non-zero, standard practice.
- All linear weights: N(0, 0.02)
- All RMSNorm γ: 1.0  (already set by RMSNorm.__init__)
- Output projection weights = Embedding weights (weight tying).
- Scaled output projections: attention o_proj and FFN out_proj are
  scaled by 1/√(2·num_layers) to prevent residual stream variance
  from growing with depth (following GPT-2 / LLaMA init practice).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from cyberslm.model.config import CyberSLMConfig, default_config
from cyberslm.model.block import DecoderBlock
from cyberslm.model.norm import RMSNorm
from cyberslm.model.rope import RotaryPositionEmbedding


class CyberSLM(nn.Module):
    """
    CyberSLM Decoder-Only Transformer.

    Parameters
    ----------
    config : CyberSLMConfig
        Validated model configuration.

    Attributes
    ----------
    config : CyberSLMConfig
    embedding : nn.Embedding
        Token embedding table, shape ``(vocab_size, hidden_dim)``.
    layers : nn.ModuleList[DecoderBlock]
        Stack of ``num_layers`` decoder blocks.
    final_norm : RMSNorm
        Applied to the residual stream after the last block.
    lm_head : nn.Linear
        Projects hidden_dim → vocab_size.  Weight tied to ``embedding``.
    """

    def __init__(self, config: CyberSLMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        # ------------------------------------------------------------------ #
        # Token embedding                                                      #
        # ------------------------------------------------------------------ #
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # ------------------------------------------------------------------ #
        # Decoder stack                                                        #
        # ------------------------------------------------------------------ #
        # One RoPE table shared by every layer (identical by construction).
        self.rope = RotaryPositionEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )
        self.layers = nn.ModuleList(
            [
                DecoderBlock(config, layer_idx=i, rope=self.rope)
                for i in range(config.num_layers)
            ]
        )

        # ------------------------------------------------------------------ #
        # Final normalisation                                                  #
        # ------------------------------------------------------------------ #
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # ------------------------------------------------------------------ #
        # Language model head                                                  #
        # ------------------------------------------------------------------ #
        # bias=False: unembedding never needs a bias term.
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # ------------------------------------------------------------------ #
        # Weight tying: lm_head.weight ≡ embedding.weight                    #
        # ------------------------------------------------------------------ #
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

        # ------------------------------------------------------------------ #
        # Parameter initialisation                                             #
        # ------------------------------------------------------------------ #
        self._init_weights()

    # ---------------------------------------------------------------------- #
    # Initialisation                                                           #
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """
        Initialise all parameters with production-quality values.

        Strategy
        --------
        - Embedding          : N(0, 0.02)
        - All Linear weights : N(0, 0.02)
        - o_proj and out_proj: scaled down by 1/√(2·L) where L = num_layers
          to stabilise the residual stream variance at initialisation.
        - All RMSNorm γ      : 1.0  (already set in RMSNorm.__init__)
        - All biases         : 0.0  (none exist in this config)
        """
        std = 0.02
        scaled_std = std / math.sqrt(2.0 * self.config.num_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                # Scaled init for residual output projections.
                if name.endswith("o_proj") or name.endswith("out_proj"):
                    nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Weight tying must be re-applied after init because _init_weights
        # initialised embedding.weight; lm_head.weight already points to the
        # same tensor (Python object reference), so no extra step needed.
        # Verify it is still tied.
        if self.config.tie_weights:
            assert self.lm_head.weight is self.embedding.weight, (
                "Weight tying broken after _init_weights"
            )

    # ---------------------------------------------------------------------- #
    # Forward pass                                                             #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_all_attn_weights: bool = False,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """
        Run the full forward pass.

        Parameters
        ----------
        input_ids : Tensor
            Long tensor of shape ``(batch, seq_len)`` with token IDs in
            ``[0, vocab_size)``.
        attention_mask : Optional[Tensor]
            Optional key-padding mask ``(batch, seq_len)`` (1=keep, 0=pad).
            Pass this when batches contain right-padded sequences so padded
            positions do not corrupt real tokens; leave ``None`` for packed,
            unpadded training batches.
        return_all_attn_weights : bool
            If True, collect and return attention weights from every layer.
            Disabled by default for training efficiency.

        Returns
        -------
        logits : Tensor
            Shape ``(batch, seq_len, vocab_size)``.  Raw (pre-softmax) scores.
        all_attn_weights : List[Optional[Tensor]]
            One entry per decoder block; each is either the attention weight
            tensor or ``None``.

        Notes
        -----
        For language model training the standard loss is:
            loss = cross_entropy(logits[:, :-1].reshape(-1, V),
                                 input_ids[:, 1:].reshape(-1))
        where we predict the next token at every position.
        """
        # ------------------------------------------------------------------ #
        # 1. Token embedding                                                   #
        # ------------------------------------------------------------------ #
        x = self.embedding(input_ids)  # (B, T, hidden_dim)

        # ------------------------------------------------------------------ #
        # 2. Decoder stack                                                     #
        # ------------------------------------------------------------------ #
        all_attn_weights: List[Optional[Tensor]] = []
        for block in self.layers:
            # use_cache=False -> `present` is None; forward() deliberately keeps
            # its (logits, attn_weights) return signature unchanged. Cached
            # decoding lives in generate() instead of overloading this method.
            x, attn_w, _ = block(
                x,
                attention_mask=attention_mask,
                return_attn_weights=return_all_attn_weights,
            )
            all_attn_weights.append(attn_w)

        # ------------------------------------------------------------------ #
        # 3. Final normalisation                                               #
        # ------------------------------------------------------------------ #
        x = self.final_norm(x)  # (B, T, hidden_dim)

        # ------------------------------------------------------------------ #
        # 4. LM head (weight-tied unembedding)                                #
        # ------------------------------------------------------------------ #
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, all_attn_weights


    # ---------------------------------------------------------------------- #
    # Cached autoregressive generation                                         #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate continuations using a KV cache.

        Why this exists
        ---------------
        The previous generation loop re-ran the whole 12-layer stack over the
        entire prefix for every single token, making decoding O(n^2) in
        sequence length. With a cache each step attends over the cached keys and
        only computes the new token, which is O(n) overall.

        Sampling is applied per row, so batched prompts are supported. Rows that
        have emitted ``eos_id`` are frozen (further tokens are forced to
        ``eos_id``) and generation stops once every row is finished.

        Parameters
        ----------
        input_ids : Tensor  ``(batch, prompt_len)`` of token ids.
        temperature : 0.0 selects greedy argmax; >0 samples.
        top_k / top_p : 0 and 1.0 respectively disable the filter.
        repetition_penalty : >1.0 divides logits of already-present tokens.
        eos_id : stop token; ``None`` means never stop early.

        Returns
        -------
        Tensor ``(batch, prompt_len + generated)`` including the prompt.
        """
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)
        max_ctx = self.config.max_seq_len

        if input_ids.size(1) >= max_ctx:
            input_ids = input_ids[:, -(max_ctx - 1):]

        caches: List[Optional[tuple]] = [None] * len(self.layers)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out = input_ids
        cur = input_ids

        for _ in range(max_new_tokens):
            if out.size(1) >= max_ctx:
                break

            h = self.embedding(cur)
            new_caches = []
            for block, layer_cache in zip(self.layers, caches):
                h, _, present = block(h, kv_cache=layer_cache, use_cache=True)
                new_caches.append(present)
            caches = new_caches
            logits = self.lm_head(self.final_norm(h))[:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(B):
                    seen = torch.unique(out[b])
                    lg = logits[b, seen]
                    # Divide positives, multiply negatives, so the penalty always
                    # pushes a token DOWN regardless of its logit's sign.
                    logits[b, seen] = torch.where(
                        lg > 0, lg / repetition_penalty, lg * repetition_penalty
                    )

            if temperature == 0.0:
                nxt = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                if top_p < 1.0:
                    srt, idx = torch.sort(logits, descending=True, dim=-1)
                    probs = torch.softmax(srt, dim=-1)
                    cum = probs.cumsum(dim=-1) - probs   # prob mass strictly before this token
                    srt = srt.masked_fill(cum > top_p, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, idx, srt)
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)

            if eos_id is not None:
                nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
                finished = finished | (nxt == eos_id)

            cur = nxt.unsqueeze(1)
            out = torch.cat([out, cur], dim=1)

            if eos_id is not None and bool(finished.all()):
                break

        return out

    # ---------------------------------------------------------------------- #
    # Convenience: next-token logits                                           #
    # ---------------------------------------------------------------------- #

    def get_next_token_logits(self, input_ids: Tensor) -> Tensor:
        """
        Return logits for the next token after the last input position.

        Parameters
        ----------
        input_ids : Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Shape ``(batch, vocab_size)``.
        """
        logits, _ = self.forward(input_ids)
        return logits[:, -1, :]  # (B, vocab_size)


# --------------------------------------------------------------------------- #
# Parameter counting                                                            #
# --------------------------------------------------------------------------- #

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters.

    Because of weight tying, lm_head.weight is counted only once
    (it shares storage with embedding.weight).

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict with keys:
        ``total``     — total parameter elements (no double-counting)
        ``trainable`` — trainable parameter elements
    """
    seen: set = set()
    total = 0
    trainable = 0
    for param in model.parameters():
        # data_ptr() is unique per underlying storage tensor.
        if param.data_ptr() in seen:
            continue
        seen.add(param.data_ptr())
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


# --------------------------------------------------------------------------- #
# Model summary                                                                 #
# --------------------------------------------------------------------------- #

def model_summary(model: CyberSLM) -> str:
    """
    Return a human-readable model summary string.

    Parameters
    ----------
    model : CyberSLM

    Returns
    -------
    str
        Formatted multi-line summary including per-component parameter counts.
    """
    cfg = model.config
    param_info = count_parameters(model)

    lines = [
        "=" * 60,
        f"  CyberSLM Model Summary",
        "=" * 60,
        f"  Architecture   : Decoder-only Transformer",
        f"  Hidden dim     : {cfg.hidden_dim}",
        f"  Num layers     : {cfg.num_layers}",
        f"  Num heads      : {cfg.num_heads}",
        f"  Head dim       : {cfg.head_dim}",
        f"  FFN hidden dim : {cfg.ffn_hidden_dim}",
        f"  Vocab size     : {cfg.vocab_size:,}",
        f"  Max seq len    : {cfg.max_seq_len:,}",
        f"  Weight tied    : {cfg.tie_weights}",
        f"  RoPE base      : {cfg.rope_base}",
        "-" * 60,
        f"  Total params   : {param_info['total']:>14,}",
        f"  Trainable      : {param_info['trainable']:>14,}",
        "-" * 60,
        "  Per-component:",
    ]

    # Embedding
    emb_p = model.embedding.weight.numel()
    lines.append(f"    Embedding              : {emb_p:>12,}")

    # Per-block breakdown (just the first block, all identical)
    block = model.layers[0]
    attn_norm_p = sum(p.numel() for p in block.attn_norm.parameters())
    attn_p      = sum(p.numel() for p in block.attn.parameters())
    ffn_norm_p  = sum(p.numel() for p in block.ffn_norm.parameters())
    ffn_p       = sum(p.numel() for p in block.ffn.parameters())
    block_total = attn_norm_p + attn_p + ffn_norm_p + ffn_p
    lines.append(f"    Decoder block (×{cfg.num_layers:2d})     : {block_total:>12,}  per block")
    lines.append(f"      attn_norm            : {attn_norm_p:>12,}")
    lines.append(f"      attention (Q/K/V/O)  : {attn_p:>12,}")
    lines.append(f"      ffn_norm             : {ffn_norm_p:>12,}")
    lines.append(f"      ffn (gate/val/out)   : {ffn_p:>12,}")
    lines.append(f"    Decoder stack total    : {block_total * cfg.num_layers:>12,}")

    # Final norm
    final_norm_p = sum(p.numel() for p in model.final_norm.parameters())
    lines.append(f"    Final RMSNorm          : {final_norm_p:>12,}")

    # LM head — note: weight tied, so 0 additional params
    lm_tied_note = " (weight-tied, no extra params)" if cfg.tie_weights else ""
    lines.append(f"    LM head                : {0:>12,}{lm_tied_note}")

    lines.append("=" * 60)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Model builder                                                                 #
# --------------------------------------------------------------------------- #

def build_model(
    config: Optional[CyberSLMConfig] = None,
    device: Optional[torch.device] = None,
) -> CyberSLM:
    """
    Build, initialise, and optionally place the CyberSLM model.

    Parameters
    ----------
    config : CyberSLMConfig, optional
        Validated config.  Uses :func:`default_config` if None.
    device : torch.device, optional
        Target device.  Stays on CPU if None.

    Returns
    -------
    CyberSLM
        Fully initialised model ready for training.
    """
    if config is None:
        config = default_config()
    else:
        config.validate()

    model = CyberSLM(config)

    if device is not None:
        model = model.to(device)

    return model
