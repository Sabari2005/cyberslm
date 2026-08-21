"""
transformer.py
==============
The CyberSLM-2 decoder-only language model.

Structure
---------
    tokens -> embedding -> [DecoderBlock] x N -> RMSNorm -> tied LM head

Initialization
--------------
Weights use N(0, 0.02) except the two projections that write into the residual
stream (attention output and FFN down), which are scaled by 1/sqrt(2*n_layers).
Without that scaling the residual stream's variance grows linearly with depth,
because each of the 2*N sublayers adds an independent contribution; the scaling
keeps the output variance roughly constant regardless of how deep the model is.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cyberslm2.configs.presets import ModelConfig
from cyberslm2.model.block import DecoderBlock
from cyberslm2.model.norm import RMSNorm

KVCache = List[Tuple[Tensor, Tensor]]


class CyberSLM2(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Sharing the matrix saves V*d params (25.2M in the 98M config) and
            # acts as a regularizer: the input and output views of a token are
            # forced to agree.
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        if config.depth_scaled_init:
            self._apply_depth_scaling()

    # -- initialization -----------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _apply_depth_scaling(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.wo.weight.mul_(scale)
                block.ffn.w_down.weight.mul_(scale)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_caches: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        """
        Parameters
        ----------
        input_ids : (B, T) int64 token ids
        attn_mask : optional (B, 1, T, S) bool mask, True == may attend.
                    None means plain causal.
        kv_caches : per-layer (k, v) tensors for incremental decoding

        Returns (logits, new_caches). Logits are (B, T, vocab_size).
        """
        x = self.embed_tokens(input_ids)

        new_caches: KVCache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, cache = block(
                x, attn_mask=attn_mask, kv_cache=layer_cache, use_cache=use_cache
            )
            if use_cache and cache is not None:
                new_caches.append(cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, (new_caches if use_cache else None)

    # -- accounting ---------------------------------------------------------

    def num_parameters(self, non_embedding: bool = False) -> int:
        """
        Count parameters. Note that a tied lm_head shares storage with the
        embedding, so a naive sum over parameters() would not double count it
        anyway -- PyTorch deduplicates shared Parameters in .parameters().
        """
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed_tokens.weight.numel()
        return total

    def summary(self) -> str:
        analytic = self.config.param_count()
        actual = self.num_parameters()
        lines = [
            f"CyberSLM2  d_model={self.config.d_model} layers={self.config.n_layers} "
            f"heads={self.config.n_heads}/{self.config.n_kv_heads}kv "
            f"ffn={self.config.ffn_hidden} vocab={self.config.vocab_size}",
            f"  analytic total : {analytic['total']:,}",
            f"  actual  total : {actual:,}",
            f"  non-embedding : {self.num_parameters(non_embedding=True):,}",
            f"  match         : {'OK' if analytic['total'] == actual else 'MISMATCH'}",
        ]
        return "\n".join(lines)

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: Optional[int] = None,
        stop_ids: Optional[List[int]] = None,
    ) -> Tensor:
        """
        Autoregressive sampling with a KV cache.

        ``stop_ids`` lets agentic decoding halt on a control token (for example
        the end of a tool call) rather than only on EOS.
        """
        self.eval()
        stop_set = set(stop_ids or [])
        if eos_id is not None:
            stop_set.add(eos_id)

        caches: Optional[KVCache] = None
        generated = input_ids

        for step in range(max_new_tokens):
            # First pass consumes the whole prompt; later passes feed one token.
            step_input = generated if caches is None else generated[:, -1:]
            logits, caches = self(step_input, kv_caches=caches, use_cache=True)
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    seen = torch.unique(generated[b])
                    scores = logits[b, seen]
                    # divide positives, multiply negatives -- both push toward 0
                    logits[b, seen] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty
                    )

            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.shape[-1])
                    kth = logits.topk(k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, sorted_idx, sorted_logits
                    )
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)
            if stop_set and next_token.item() in stop_set and generated.shape[0] == 1:
                break

        return generated
