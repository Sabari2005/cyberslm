# data/__init__.py  (inference-only subset)
#
# The full package also re-exports the dataset loader, validator, collator and
# loss masking. Those are training-only and are deliberately not shipped here,
# so importing them would fail. Only the prompt formatter is needed to build a
# prompt the model recognises.
from data.prompt_formatter import PromptFormatter, Tokenizer, IGNORE_INDEX

__all__ = ["PromptFormatter", "Tokenizer", "IGNORE_INDEX"]
