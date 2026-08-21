"""
special_tokens.py
=================
The control-token protocol for chat, explicit reasoning, and tool calling.

Everything here is a *real token id*, never a literal string in the text. That
distinction is the single most important lesson carried over from the v1 model,
where "</s>" was written as characters into the prompt: the tokenizer split it
into ordinary pieces, the model never saw a true EOS, and generation could not
stop. Ids are unambiguous and one embedding each; strings are neither.

Layout
------
Ids 0-3 are SentencePiece's built-in control slots. Ids 4+ are declared as
``user_defined_symbols`` when training the tokenizer so SentencePiece emits them
atomically and never splits them.

Conversation grammar
--------------------
    <bos>
    <|system|>   system prompt        <|end|>
    <|user|>     user turn            <|end|>
    <|assistant|>
        <|think|>  private reasoning  <|/think|>     (optional, loss-masked out
                                                      at inference, trained on)
        visible answer
    <|end|>
    <eos>

Tool calling
------------
    <|assistant|><|tool_call|>{"name": "...", "arguments": {...}}<|/tool|><|end|>
    <|tool_result|>{"ok": true, "output": "..."}<|/tool|>
    <|assistant|> final answer <|end|>

The payload between <|tool_call|> and <|/tool|> is plain JSON, so the model
learns one syntax it already sees constantly in the pretraining corpus rather
than a bespoke format it must learn from SFT data alone.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- SentencePiece built-ins (ids fixed by the trainer flags) ---------------
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# --- user-defined control tokens, in the exact order passed to the trainer --
# Order matters: it determines the ids, and the ids are baked into every
# tokenized corpus. Append new tokens at the END, never insert in the middle.
CONTROL_TOKENS: list[str] = [
    "<|system|>",       # 4
    "<|user|>",         # 5
    "<|assistant|>",    # 6
    "<|end|>",          # 7   end of a single turn
    "<|think|>",        # 8   open private reasoning
    "<|/think|>",       # 9   close private reasoning
    "<|tool_list|>",    # 10  declares the available tool schemas
    "<|tool_call|>",    # 11  model -> environment
    "<|tool_result|>",  # 12  environment -> model
    "<|/tool|>",        # 13  closes a tool payload
    "<|code|>",         # 14  fenced executable code
    "<|/code|>",        # 15
]

FIRST_CONTROL_ID = 4
CONTROL_IDS: dict[str, int] = {
    tok: FIRST_CONTROL_ID + i for i, tok in enumerate(CONTROL_TOKENS)
}

# Convenience constants
SYSTEM_ID = CONTROL_IDS["<|system|>"]
USER_ID = CONTROL_IDS["<|user|>"]
ASSISTANT_ID = CONTROL_IDS["<|assistant|>"]
END_ID = CONTROL_IDS["<|end|>"]
THINK_ID = CONTROL_IDS["<|think|>"]
THINK_END_ID = CONTROL_IDS["<|/think|>"]
TOOL_LIST_ID = CONTROL_IDS["<|tool_list|>"]
TOOL_CALL_ID = CONTROL_IDS["<|tool_call|>"]
TOOL_RESULT_ID = CONTROL_IDS["<|tool_result|>"]
TOOL_END_ID = CONTROL_IDS["<|/tool|>"]
CODE_ID = CONTROL_IDS["<|code|>"]
CODE_END_ID = CONTROL_IDS["<|/code|>"]

# Reserve headroom so a future token can be added without shifting the
# vocabulary or invalidating an already-tokenized corpus.
N_RESERVED = 32

# Decoding should stop on any of these when generating a single assistant turn.
STOP_IDS = [EOS_ID, END_ID]

# Tokens that must never be sampled by the model.
FORBIDDEN_SAMPLING_IDS = [PAD_ID, UNK_ID, BOS_ID, TOOL_RESULT_ID]

IGNORE_INDEX = -100  # CrossEntropyLoss default for "do not train on this token"


@dataclass(frozen=True)
class SpecialTokens:
    """Resolved ids, validated against an actual tokenizer at load time."""

    pad_id: int = PAD_ID
    unk_id: int = UNK_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID

    @staticmethod
    def validate(sp) -> None:
        """
        Check a loaded SentencePiece processor really matches this protocol.

        Call this once at startup. A silent id mismatch between the tokenizer
        and this module corrupts every label in the dataset while still
        producing plausible-looking loss curves, so failing loudly here is
        worth the two lines it costs.
        """
        problems = []
        if sp.pad_id() != PAD_ID:
            problems.append(f"pad_id={sp.pad_id()} expected {PAD_ID}")
        if sp.unk_id() != UNK_ID:
            problems.append(f"unk_id={sp.unk_id()} expected {UNK_ID}")
        if sp.bos_id() != BOS_ID:
            problems.append(f"bos_id={sp.bos_id()} expected {BOS_ID}")
        if sp.eos_id() != EOS_ID:
            problems.append(f"eos_id={sp.eos_id()} expected {EOS_ID}")

        for tok, expected in CONTROL_IDS.items():
            got = sp.piece_to_id(tok)
            if got != expected:
                problems.append(f"{tok}: id={got} expected {expected}")

        if problems:
            raise ValueError(
                "Tokenizer does not match the CyberSLM-2 token protocol:\n  "
                + "\n  ".join(problems)
                + "\nRetrain it with scripts/train_tokenizer_v2.py."
            )
