"""
CyberSLM SFT — Conversation Template
======================================
Defines and renders the canonical chat template used throughout training
and inference.

This module is the single source of truth for how messages are serialised
into a string.  The ``PromptFormatter`` delegates format-specific string
construction here; nothing else should hardcode role strings or delimiters.

Template Layout (single-turn)
------------------------------

    <s>### Instruction:\n{instruction}\n\n### Response:\n{response}</s>

With optional context (alpaca "input" field):

    <s>### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}</s>

Chat template (multi-turn):

    <s>System: {system}\n\n### User:\n{user}\n\n### Assistant:\n{assistant}</s>
    \n\n### User:\n{user2}\n\n### Assistant:\n{assistant2}</s>

Special tokens / delimiters are configurable via ``TemplateConfig`` but
the defaults above are used throughout the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from configs.sft_config import TemplateConfig


# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------

ROLE_SYSTEM    = "system"
ROLE_USER      = "user"
ROLE_ASSISTANT = "assistant"

# Rendered section headers used in the prompt string
_INSTRUCTION_HEADER = "### Instruction:"
_INPUT_HEADER       = "### Input:"
_RESPONSE_HEADER    = "### Response:"
_USER_HEADER        = "### User:"
_ASSISTANT_HEADER   = "### Assistant:"
_SYSTEM_PREFIX      = "System:"
# BOS/EOS are handled as real token ids (bos_id=2, eos_id=3), never as literal
# strings. The template therefore injects no special-token text; callers
# prepend ``tokenizer.bos_id`` to the encoded prompt.
_BOS                = ""


# ---------------------------------------------------------------------------
# Rendered segment
# ---------------------------------------------------------------------------

@dataclass
class RenderedSegment:
    """
    One logical piece of a rendered conversation string.

    Attributes
    ----------
    text:           The raw string for this segment.
    is_loss_target: Whether this segment should contribute to the loss.
                    True only for assistant response bodies.
    char_start:     Start character offset in the full assembled string
                    (populated by ``ConversationTemplate.render``).
    char_end:       Exclusive end character offset (populated by ``render``).
    """
    text:           str
    is_loss_target: bool
    char_start:     int = 0
    char_end:       int = 0


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

class ConversationTemplate:
    """
    Renders instruction samples and multi-turn conversations to strings,
    returning segment metadata that the loss masker uses to identify which
    character ranges correspond to assistant responses.

    Parameters
    ----------
    cfg:
        ``TemplateConfig`` — controls the response prefix, EOS string, and
        whether to strip field whitespace.
    """

    def __init__(self, cfg: Optional[TemplateConfig] = None) -> None:
        self.cfg = cfg or TemplateConfig()

    # ------------------------------------------------------------------
    # Alpaca / instruction format
    # ------------------------------------------------------------------

    def render_alpaca(
        self,
        instruction: str,
        output:      str,
        input_text:  str = "",
        for_inference: bool = False,
    ) -> Tuple[str, List[RenderedSegment]]:
        """
        Render a single-turn instruction sample.

        Parameters
        ----------
        instruction:   Task description.
        output:        Expected assistant response.
        input_text:    Optional context (empty → section omitted).
        for_inference: If True, the response body is omitted (generation).

        Returns
        -------
        (full_text, segments)
            *full_text*  — The complete string fed to the tokeniser.
            *segments*   — List of ``RenderedSegment`` with char offsets set.
        """
        if self.cfg.strip_fields:
            instruction = instruction.strip()
            input_text  = input_text.strip()
            output      = output.strip()

        segments: List[RenderedSegment] = []
        cursor = 0

        def _add(seg: RenderedSegment) -> None:
            nonlocal cursor
            seg.char_start = cursor
            seg.char_end   = cursor + len(seg.text)
            cursor         = seg.char_end
            segments.append(seg)

        # BOS + instruction
        if input_text:
            prompt_body = (
                f"{_BOS}{_INSTRUCTION_HEADER}\n{instruction}\n\n"
                f"{_INPUT_HEADER}\n{input_text}\n\n"
                f"{_RESPONSE_HEADER}\n"
            )
        else:
            prompt_body = (
                f"{_BOS}{_INSTRUCTION_HEADER}\n{instruction}\n\n"
                f"{_RESPONSE_HEADER}\n"
            )

        _add(RenderedSegment(text=prompt_body, is_loss_target=False))

        # Response (training only)
        if not for_inference:
            response_body = output + self.cfg.eos_string
            _add(RenderedSegment(text=response_body, is_loss_target=True))

        full_text = "".join(s.text for s in segments)
        return full_text, segments

    # ------------------------------------------------------------------
    # Conversation / multi-turn format
    # ------------------------------------------------------------------

    def render_conversation(
        self,
        messages:      List[dict],
        for_inference: bool = False,
    ) -> Tuple[str, List[RenderedSegment]]:
        """
        Render a multi-turn message list.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.  A leading
            ``system`` role is rendered as a header before the first turn.
        for_inference:
            If True, the *final* assistant turn body is omitted so the model
            can generate it.

        Returns
        -------
        (full_text, segments)
        """
        segments: List[RenderedSegment] = []
        cursor = 0

        def _add(seg: RenderedSegment) -> None:
            nonlocal cursor
            seg.char_start = cursor
            seg.char_end   = cursor + len(seg.text)
            cursor         = seg.char_end
            segments.append(seg)

        # ---- separate system prompt ----
        system_content = ""
        body_messages  = messages
        if messages and messages[0]["role"] == ROLE_SYSTEM:
            system_content = messages[0]["content"]
            if self.cfg.strip_fields:
                system_content = system_content.strip()
            body_messages  = messages[1:]

        # ---- BOS + optional system ----
        if system_content:
            header_text = f"{_BOS}{_SYSTEM_PREFIX} {system_content}\n\n"
        else:
            header_text = _BOS
        _add(RenderedSegment(text=header_text, is_loss_target=False))

        # ---- turns ----
        i = 0
        turns = body_messages
        n     = len(turns)

        while i < n:
            role    = turns[i]["role"]
            content = turns[i]["content"]
            if self.cfg.strip_fields:
                content = content.strip()

            if role == ROLE_USER:
                user_block = f"{_USER_HEADER}\n{content}\n\n"
                _add(RenderedSegment(text=user_block, is_loss_target=False))

                # Check for following assistant turn
                if i + 1 < n and turns[i + 1]["role"] == ROLE_ASSISTANT:
                    asst_content = turns[i + 1]["content"]
                    if self.cfg.strip_fields:
                        asst_content = asst_content.strip()

                    is_final = (i + 2 >= n)

                    # The assistant prefix is always part of the prompt (masked)
                    asst_prefix = f"{_ASSISTANT_HEADER}\n"
                    _add(RenderedSegment(text=asst_prefix, is_loss_target=False))

                    if is_final and for_inference:
                        # Omit the response body entirely for generation
                        pass
                    else:
                        asst_body = asst_content + self.cfg.eos_string
                        _add(RenderedSegment(text=asst_body, is_loss_target=True))
                        # Inter-turn separator
                        _add(RenderedSegment(text="\n\n", is_loss_target=False))

                    i += 2  # consumed user + assistant
                else:
                    # Dangling user turn — no assistant reply, skip
                    i += 1

            elif role == ROLE_ASSISTANT:
                # Orphaned assistant turn (no preceding user turn)
                asst_prefix = f"{_ASSISTANT_HEADER}\n"
                _add(RenderedSegment(text=asst_prefix, is_loss_target=False))

                content_body = content + self.cfg.eos_string
                _add(RenderedSegment(text=content_body, is_loss_target=True))
                _add(RenderedSegment(text="\n\n", is_loss_target=False))
                i += 1

            else:
                # system mid-conversation: treat as informational (masked)
                sys_block = f"{_SYSTEM_PREFIX} {content}\n\n"
                _add(RenderedSegment(text=sys_block, is_loss_target=False))
                i += 1

        full_text = "".join(s.text for s in segments)
        return full_text, segments

    # ------------------------------------------------------------------
    # Convenience: extract loss-target char spans
    # ------------------------------------------------------------------

    def get_response_spans(
        self,
        segments: List[RenderedSegment],
    ) -> List[Tuple[int, int]]:
        """
        Return ``(char_start, char_end)`` pairs for all loss-target segments.

        Used by the tokeniser boundary logic in ``PromptFormatter``.
        """
        return [
            (s.char_start, s.char_end)
            for s in segments
            if s.is_loss_target
        ]

    # ------------------------------------------------------------------
    # Render from raw sample (dispatch)
    # ------------------------------------------------------------------

    def render(
        self,
        sample: dict,
        for_inference: bool = False,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Auto-dispatch: render any normalised sample and return
        ``(full_text, response_char_spans)``.

        Parameters
        ----------
        sample:        Normalised dict from the dataset loader.
        for_inference: Passed through to the format-specific renderer.
        """
        if "messages" in sample:
            full_text, segments = self.render_conversation(
                sample["messages"],
                for_inference=for_inference,
            )
        else:
            full_text, segments = self.render_alpaca(
                instruction=sample["instruction"],
                output=sample.get("output", ""),
                input_text=sample.get("input", ""),
                for_inference=for_inference,
            )

        spans = self.get_response_spans(segments)
        return full_text, spans
