"""
CyberSLM SFT — Data Collator
=============================
Converts a list of ``(input_ids, labels)`` pairs (produced by
``PromptFormatter``) into padded batch tensors ready for the model.

Responsibilities
----------------
* Pad ``input_ids`` to the longest sequence in the batch (right-pad with
  ``pad_id``).
* Pad ``labels`` with ``IGNORE_INDEX`` (-100) so padding positions never
  contribute to the loss.
* Build ``attention_mask`` (1 for real tokens, 0 for padding).
* Optionally truncate sequences that exceed ``max_seq_len`` before padding.

The collator is stateless and can be passed directly to
``torch.utils.data.DataLoader`` as the ``collate_fn`` argument.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from data.prompt_formatter import IGNORE_INDEX


class SFTCollator:
    """
    Callable collator for SFT ``(input_ids, labels)`` pairs.

    Parameters
    ----------
    pad_id:
        Token id used to right-pad ``input_ids`` to the batch maximum length.
    max_seq_len:
        Hard upper bound on sequence length.  Sequences longer than this are
        truncated *before* padding.  Set to a large value (e.g. 4096) if you
        trust the formatter to have already truncated.
    """

    def __init__(self, pad_id: int, max_seq_len: int = 4096) -> None:
        self.pad_id      = pad_id
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    def __call__(
        self,
        batch: List[Tuple[List[int], List[int]]],
    ) -> dict[str, Tensor]:
        """
        Collate a list of (input_ids, labels) pairs into batch tensors.

        Parameters
        ----------
        batch:
            Each element is ``(input_ids, labels)`` — both plain Python lists
            of ints with the same length.

        Returns
        -------
        dict with keys:
            ``input_ids``      — shape (B, T)  dtype int64
            ``labels``         — shape (B, T)  dtype int64  (-100 for masked)
            ``attention_mask`` — shape (B, T)  dtype int64  (1/0)
        """
        # ---- truncate ----
        truncated = [
            (ids[: self.max_seq_len], lbl[: self.max_seq_len])
            for ids, lbl in batch
        ]

        max_len = max(len(ids) for ids, _ in truncated)

        input_ids_list:  List[List[int]] = []
        labels_list:     List[List[int]] = []
        attn_mask_list:  List[List[int]] = []

        for ids, lbl in truncated:
            seq_len = len(ids)
            pad_len = max_len - seq_len

            input_ids_list.append(ids + [self.pad_id]    * pad_len)
            labels_list.append(   lbl + [IGNORE_INDEX]   * pad_len)
            attn_mask_list.append([1]  * seq_len + [0]   * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids_list,  dtype=torch.long),
            "labels":         torch.tensor(labels_list,     dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_list,  dtype=torch.long),
        }
