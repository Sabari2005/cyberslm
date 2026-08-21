"""
CyberSLM SFT — Prompt Formatter
================================
Converts normalised raw samples into tokenised ``(input_ids, labels)``
pairs ready for the data collator.

Tokenisation strategy (segment-based)
-------------------------------------
A sample is decomposed into an ordered list of ``(text, is_loss_target)``
segments (see ``ConversationTemplate`` for the canonical layout). Each
segment is encoded independently and the resulting token id lists are
concatenated to form ``input_ids``. ``labels`` mirror ``input_ids`` but with
non-target (prompt) positions set to ``IGNORE_INDEX`` (-100).

Why segment-based (and not char-offset) masking
------------------------------------------------
Locating the response boundary by re-encoding a prefix string and comparing
token counts is fragile: with ``add_dummy_prefix`` (and BPE merges across the
boundary) ``len(encode(prefix))`` need not equal the number of full-sequence
tokens covering that prefix, so the mask can drift by 1-2 tokens per turn.
Encoding each segment and concatenating gives an **exact** boundary because
``input_ids`` and ``labels`` are built from the very same token lists.

Special tokens (critical)
--------------------------
Special tokens are referenced by **id**, never as literal strings. The real
SentencePiece control ids are ``<bos>=2`` and ``<eos>=3``. Every assistant
response is terminated with ``eos_id`` as a genuine token and that ``eos_id``
is **left unmasked** so the model learns to stop. ``bos_id`` is prepended once
at the start of the sequence (masked). This replaces the previous, broken
approach of appending the literal characters ``"</s>"`` (which SentencePiece
encodes as ``<``,``/``,``s``,``>`` — never id 3), which prevented the model
from ever learning an end-of-sequence signal.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from configs.sft_config import SFTConfig, TemplateConfig

logger = logging.getLogger(__name__)

# Tokens with this label are excluded from the cross-entropy loss.
IGNORE_INDEX: int = -100

# One segment of the rendered prompt: (text, is_loss_target).
Segment = Tuple[str, bool]


# ---------------------------------------------------------------------------
# Segment builders (string level, EOS handled at token level downstream)
# ---------------------------------------------------------------------------

def build_alpaca_segments(
    instruction: str,
    output: str,
    input_text: str = "",
    tmpl: Optional[TemplateConfig] = None,
    include_response: bool = True,
) -> List[Segment]:
    """
    Build the ordered ``(text, is_loss_target)`` segments for an alpaca sample.

    The response text is a loss target; everything else (instruction, input,
    the response header) is masked. No EOS string is injected here — the
    end-of-sequence token is appended as a real token id by
    :func:`encode_segments`.
    """
    if tmpl is None:
        tmpl = TemplateConfig()

    if tmpl.strip_fields:
        instruction = instruction.strip()
        input_text = input_text.strip()
        output = output.strip()

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"{tmpl.response_prefix}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"{tmpl.response_prefix}"
        )

    segments: List[Segment] = [(prompt, False)]
    if include_response:
        segments.append((output, True))
    return segments


def build_conversation_segments(
    messages: List[dict],
    tmpl: Optional[TemplateConfig] = None,
    include_final_response: bool = True,
) -> List[Segment]:
    """
    Build ordered ``(text, is_loss_target)`` segments for a multi-turn
    conversation. Only assistant response bodies are loss targets.
    """
    if tmpl is None:
        tmpl = TemplateConfig()

    def _clean(s: str) -> str:
        return s.strip() if tmpl.strip_fields else s

    segments: List[Segment] = []

    # Optional leading system message.
    body = messages
    if messages and messages[0].get("role") == "system":
        sys_content = _clean(messages[0]["content"])
        if sys_content:
            segments.append((f"System: {sys_content}\n\n", False))
        body = messages[1:]

    i = 0
    n = len(body)
    while i < n:
        role = body[i]["role"]
        content = _clean(body[i]["content"])

        if role == "user":
            segments.append((f"### User:\n{content}\n\n", False))
            # Pair with the following assistant turn, if present.
            if i + 1 < n and body[i + 1]["role"] == "assistant":
                asst = _clean(body[i + 1]["content"])
                is_last = (i + 2 >= n)
                segments.append(("### Assistant:\n", False))
                if not (is_last and not include_final_response):
                    segments.append((asst, True))
                    segments.append(("\n\n", False))
                i += 2
            else:
                # Dangling user turn with no assistant reply -- the normal
                # generation case. The assistant header MUST still be emitted:
                # it is the cue the model was trained to continue from. Omitting
                # it (the previous behaviour) handed the model a prompt shaped
                # unlike anything in its training distribution.
                segments.append(("### Assistant:\n", False))
                i += 1

        elif role == "assistant":
            asst = content
            segments.append(("### Assistant:\n", False))
            segments.append((asst, True))
            segments.append(("\n\n", False))
            i += 1

        else:
            # Mid-conversation system message: treat as masked context.
            segments.append((f"### System:\n{content}\n\n", False))
            i += 1

    return segments


# ---------------------------------------------------------------------------
# Tokeniser wrapper
# ---------------------------------------------------------------------------

class Tokenizer:
    """
    Thin wrapper around a SentencePiece model that exposes only what the
    formatter and collator need.

    SentencePiece is not imported at module level so the formatter module can
    be imported in unit tests without requiring sentencepiece to be installed.
    """

    def __init__(self, model_path: str) -> None:
        try:
            import sentencepiece as spm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is required for the tokenizer. "
                "Install with:  pip install sentencepiece"
            ) from exc

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)

        self.bos_id: int = self._sp.bos_id()
        self.eos_id: int = self._sp.eos_id()
        self.pad_id: int = self._sp.pad_id()
        self.vocab_size: int = self._sp.GetPieceSize()

        # Fail loudly if the tokenizer's control ids do not match the ids the
        # rest of the pipeline (and the base model config) assume. A silent
        # mismatch here corrupts training and breaks generation stopping.
        if self.eos_id < 0:
            raise ValueError(
                "Tokenizer has no EOS id (eos_id < 0). The SFT pipeline "
                "requires a real end-of-sequence token."
            )

    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids: List[int] = self._sp.Encode(text, out_type=int)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._sp.Decode(ids)

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Segment → token id encoding
# ---------------------------------------------------------------------------

def encode_segments(
    segments: List[Segment],
    tokenizer: Tokenizer,
    max_seq_len: int,
    add_bos: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Encode ``(text, is_loss_target)`` segments into ``(input_ids, labels)``.

    * ``input_ids`` is the concatenation of each segment's token ids.
    * A real ``eos_id`` token is appended immediately after every loss-target
      (assistant) segment and is itself a loss target — the model must learn
      to emit it.
    * ``bos_id`` is prepended once at the start (masked) when ``add_bos``.
    * ``labels`` equal ``input_ids`` on target positions and ``IGNORE_INDEX``
      elsewhere.

    The sequence is truncated to ``max_seq_len`` tokens.
    """
    input_ids: List[int] = []
    labels: List[int] = []

    if add_bos and tokenizer.bos_id is not None and tokenizer.bos_id >= 0:
        input_ids.append(tokenizer.bos_id)
        labels.append(IGNORE_INDEX)

    for text, is_target in segments:
        ids = tokenizer.encode(text)
        input_ids.extend(ids)
        labels.extend(ids if is_target else [IGNORE_INDEX] * len(ids))
        if is_target:
            # Terminate the assistant turn with a learned EOS token.
            input_ids.append(tokenizer.eos_id)
            labels.append(tokenizer.eos_id)

    return input_ids[:max_seq_len], labels[:max_seq_len]


# ---------------------------------------------------------------------------
# Public API — PromptFormatter
# ---------------------------------------------------------------------------

class PromptFormatter:
    """
    Converts a normalised raw sample into a tokenised ``(input_ids, labels)``
    pair, applying loss-masking so only assistant responses are trained on.

    Parameters
    ----------
    cfg:
        Master ``SFTConfig`` (for template and data settings).
    tokenizer:
        Initialised ``Tokenizer`` instance.
    """

    def __init__(self, cfg: SFTConfig, tokenizer: Tokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.tmpl = cfg.template
        self.max_len = cfg.data.max_seq_len

    # ------------------------------------------------------------------
    def format(self, sample: dict) -> Optional[Tuple[List[int], List[int]]]:
        """
        Format a single normalised sample into ``(input_ids, labels)``.

        Returns ``None`` when the sample produces no trainable (non-masked)
        tokens — e.g. the response was truncated away.
        """
        if "messages" in sample:
            segments = build_conversation_segments(
                sample["messages"], tmpl=self.tmpl, include_final_response=True
            )
        else:
            segments = build_alpaca_segments(
                instruction=sample["instruction"],
                output=sample["output"],
                input_text=sample.get("input", ""),
                tmpl=self.tmpl,
                include_response=True,
            )

        # Nothing to train on if there are no target segments at all.
        if not any(is_target for _, is_target in segments):
            logger.debug("No assistant/response segments — skipping sample")
            return None

        input_ids, labels = encode_segments(
            segments, self.tokenizer, self.max_len, add_bos=True
        )

        if not input_ids or not any(l != IGNORE_INDEX for l in labels):
            logger.debug(
                "No response tokens remain after truncation at %d — skipping",
                self.max_len,
            )
            return None

        return input_ids, labels

    # ------------------------------------------------------------------
    def format_for_inference(self, sample: dict) -> List[int]:
        """
        Format a sample for *generation*: prompt only, no response body and no
        trailing EOS, with ``bos_id`` prepended.
        """
        if "messages" in sample:
            segments = build_conversation_segments(
                sample["messages"], tmpl=self.tmpl, include_final_response=False
            )
        else:
            segments = build_alpaca_segments(
                instruction=sample["instruction"],
                output=sample.get("output", ""),
                input_text=sample.get("input", ""),
                tmpl=self.tmpl,
                include_response=False,
            )
        input_ids, _ = encode_segments(
            segments, self.tokenizer, self.max_len, add_bos=True
        )
        return input_ids
