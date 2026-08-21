"""
packing.py
==========
Sequence packing and document-aware attention masking.

The problem
-----------
Pretraining corpora are millions of short documents. Two naive options both
waste the budget:

  * Pad every document to seq_len -> most of every batch is padding, and on this
    corpus (mean ~307 tokens/doc vs a 2048 window) that would throw away ~85%
    of the compute.
  * Concatenate everything and cut at fixed strides -> no padding, but tokens at
    the start of document B can attend to document A, so the model learns
    correlations across a boundary that carries no information.

The fix
-------
Pack documents back-to-back and give attention a *block-diagonal* mask so each
token only sees tokens from its own document. Full density, no cross-document
leakage.

Recovering boundaries
---------------------
No side-car index is needed. Every document in the token stream terminates with
<eos>, so a token starts a new document iff the previous token was <eos>:

    doc_id = cumsum(shift_right(tokens == EOS))

That is exact, costs one pass, and works on the existing .bin format.

Memory note
-----------
The materialized mask is (B, 1, T, T) bool: at B=8, T=2048 that is 33.5 MB,
which is fine. If you move to T=8192 (~537 MB) switch to torch's
``flex_attention`` BlockMask, which represents the same mask sparsely. The
helper below is deliberately the simple correct version.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from cyberslm2.data.special_tokens import EOS_ID


def build_doc_ids(tokens: Tensor, eos_id: int = EOS_ID) -> Tensor:
    """
    Assign each position an integer document index.

    tokens: (B, T) -> doc_ids: (B, T), monotonically non-decreasing per row.
    The token *after* an <eos> begins the next document; the <eos> itself
    belongs to the document it terminates.
    """
    is_eos = (tokens == eos_id).to(torch.int32)
    # shift right by one: position t opens a new doc if position t-1 was <eos>
    starts = torch.zeros_like(is_eos)
    starts[:, 1:] = is_eos[:, :-1]
    return starts.cumsum(dim=1)


def build_doc_causal_mask(
    doc_ids: Tensor,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """
    Block-diagonal causal mask.

    Returns (B, 1, T, T) where entry [b, 0, i, j] is True when query i may
    attend to key j, i.e. when j <= i AND both positions are in the same
    document. Shaped for direct use as SDPA's ``attn_mask``.
    """
    B, T = doc_ids.shape
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]          # (B, T, T)
    causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
    mask = same_doc & causal
    mask = mask.unsqueeze(1)                                        # (B, 1, T, T)

    if dtype is torch.bool:
        return mask
    # additive form: 0 where allowed, -inf where blocked
    return torch.zeros_like(mask, dtype=dtype).masked_fill(
        ~mask, torch.finfo(dtype).min
    )


def build_padding_causal_mask(
    attention_mask: Tensor,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """
    Causal mask combined with a key-padding mask, for right-padded SFT batches.

    attention_mask: (B, T) with 1 for real tokens and 0 for padding.
    """
    B, T = attention_mask.shape
    keep = attention_mask.to(torch.bool)
    causal = torch.ones(T, T, dtype=torch.bool, device=attention_mask.device).tril()
    mask = causal[None, :, :] & keep[:, None, :]                   # block padded keys
    mask = mask.unsqueeze(1)

    if dtype is torch.bool:
        return mask
    return torch.zeros_like(mask, dtype=dtype).masked_fill(
        ~mask, torch.finfo(dtype).min
    )


def pack_documents(
    docs: list[list[int]],
    seq_len: int,
    eos_id: int = EOS_ID,
    drop_remainder: bool = True,
) -> list[list[int]]:
    """
    Greedily concatenate documents into fixed-length windows.

    Each document is assumed to already carry its terminating <eos>; if it does
    not, one is appended so boundaries stay recoverable. Documents longer than
    seq_len are split across consecutive windows rather than discarded, which
    matters here because long CVE writeups and code files are exactly the
    high-value samples.
    """
    windows: list[list[int]] = []
    buf: list[int] = []

    for doc in docs:
        toks = doc if (doc and doc[-1] == eos_id) else [*doc, eos_id]
        buf.extend(toks)
        while len(buf) >= seq_len:
            windows.append(buf[:seq_len])
            buf = buf[seq_len:]

    if buf and not drop_remainder:
        buf.extend([eos_id] * (seq_len - len(buf)))
        windows.append(buf)

    return windows


def shift_for_causal_lm(
    input_ids: Tensor,
    labels: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor]:
    """
    Produce the (input, target) pair for next-token prediction.

    Returns inputs[..., :-1] and targets[..., 1:]. Doing the shift in one place
    avoids the classic off-by-one where loss is computed against the token the
    model was actually given.
    """
    tgt = labels if labels is not None else input_ids
    return input_ids[..., :-1].contiguous(), tgt[..., 1:].contiguous()
