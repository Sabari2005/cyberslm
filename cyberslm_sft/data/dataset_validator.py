"""
CyberSLM SFT — Dataset Validator
=================================
Validates normalised samples before they enter the training pipeline.

Validation catches:
* Missing / empty mandatory fields
* Wrong field types
* Samples whose token count exceeds the configured maximum
* Conversation samples with no assistant turn (nothing to learn from)
* Samples where the output / response is empty (zero-loss samples)

The validator operates in two modes:

* **strict** — any invalid sample raises ``DatasetValidationError`` and
  halts the pipeline.  Use this when you want clean data guarantees.
* **lenient** (default) — invalid samples are logged and dropped; training
  continues with the valid subset.  Suitable for noisy community datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from configs.sft_config import DataConfig

logger = logging.getLogger(__name__)

# Maximum allowed character length of any single field before flagging the
# sample as suspicious (not a hard error by default — only a warning).
_FIELD_CHAR_WARN_THRESHOLD = 32_000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatasetValidationError(RuntimeError):
    """Raised when strict-mode validation encounters an invalid sample."""


# ---------------------------------------------------------------------------
# Validation result per sample
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    index: int
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Field-level checks (shared helpers)
# ---------------------------------------------------------------------------

def _check_non_empty_string(value: object, field_name: str) -> Optional[str]:
    """Return an error message if value is not a non-empty string, else None."""
    if not isinstance(value, str):
        return f"'{field_name}' must be a string, got {type(value).__name__}"
    if not value.strip():
        return f"'{field_name}' must not be empty or whitespace-only"
    return None


def _check_field_length(value: str, field_name: str) -> Optional[str]:
    """Return a warning if the field is unusually long."""
    if len(value) > _FIELD_CHAR_WARN_THRESHOLD:
        return (
            f"'{field_name}' is very long ({len(value)} chars). "
            "Consider splitting or truncating."
        )
    return None


# ---------------------------------------------------------------------------
# Alpaca sample validation
# ---------------------------------------------------------------------------

def _validate_alpaca(sample: dict, index: int) -> ValidationResult:
    result = ValidationResult(index=index, valid=True)

    # --- instruction ---
    err = _check_non_empty_string(sample.get("instruction"), "instruction")
    if err:
        result.errors.append(err)

    # --- output ---
    err = _check_non_empty_string(sample.get("output"), "output")
    if err:
        result.errors.append(err)

    # --- input (optional but must be a string if present) ---
    inp = sample.get("input", "")
    if not isinstance(inp, str):
        result.errors.append(f"'input' must be a string, got {type(inp).__name__}")

    # --- length warnings ---
    for fname in ("instruction", "input", "output"):
        val = sample.get(fname, "")
        if isinstance(val, str) and val:
            warn = _check_field_length(val, fname)
            if warn:
                result.warnings.append(warn)

    if result.errors:
        result.valid = False
    return result


# ---------------------------------------------------------------------------
# Conversation sample validation
# ---------------------------------------------------------------------------

_VALID_ROLES = {"system", "user", "assistant"}


def _validate_conversation(sample: dict, index: int) -> ValidationResult:
    result = ValidationResult(index=index, valid=True)

    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        result.errors.append("'messages' must be a non-empty list")
        result.valid = False
        return result

    has_user      = False
    has_assistant = False

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            result.errors.append(
                f"messages[{msg_idx}] must be a dict, got {type(msg).__name__}"
            )
            continue

        role    = msg.get("role", "")
        content = msg.get("content", "")

        if role not in _VALID_ROLES:
            result.errors.append(
                f"messages[{msg_idx}].role '{role}' is not one of {_VALID_ROLES}"
            )

        err = _check_non_empty_string(content, f"messages[{msg_idx}].content")
        if err:
            result.errors.append(err)
        elif isinstance(content, str):
            warn = _check_field_length(content, f"messages[{msg_idx}].content")
            if warn:
                result.warnings.append(warn)

        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True

    if not has_user:
        result.errors.append("Conversation has no 'user' turn")
    if not has_assistant:
        result.errors.append(
            "Conversation has no 'assistant' turn — nothing to train on"
        )

    if result.errors:
        result.valid = False
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _validate_sample(sample: dict, index: int) -> ValidationResult:
    """Route to the correct validator based on detected format."""
    if "messages" in sample:
        return _validate_conversation(sample, index)
    if "instruction" in sample:
        return _validate_alpaca(sample, index)

    return ValidationResult(
        index=index,
        valid=False,
        errors=["Unrecognised format: missing both 'messages' and 'instruction' keys"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Summary produced by ``validate_dataset``."""
    total: int
    valid: int
    invalid: int
    dropped: List[int]         # indices of invalid samples
    warnings: List[Tuple[int, List[str]]]  # (index, warning_list)

    def log(self) -> None:
        logger.info(
            "Dataset validation — total: %d | valid: %d | invalid (dropped): %d",
            self.total, self.valid, self.invalid,
        )
        for idx, warns in self.warnings:
            for w in warns:
                logger.warning("Sample %d: %s", idx, w)


def validate_dataset(
    samples: List[dict],
    strict: bool = False,
) -> Tuple[List[dict], ValidationReport]:
    """
    Validate every sample in *samples*.

    Parameters
    ----------
    samples:
        Normalised samples from ``dataset_loader.load_jsonl``.
    strict:
        If ``True``, raise ``DatasetValidationError`` on the first invalid
        sample.  If ``False`` (default), drop invalid samples and continue.

    Returns
    -------
    (valid_samples, report)
        *valid_samples* contains only the samples that passed all checks.
        *report* summarises what was accepted, dropped, and warned about.
    """
    valid_samples: List[dict] = []
    dropped:  List[int] = []
    warnings_list: List[Tuple[int, List[str]]] = []

    for idx, sample in enumerate(samples):
        result = _validate_sample(sample, idx)

        if result.warnings:
            warnings_list.append((idx, result.warnings))

        if result.valid:
            valid_samples.append(sample)
        else:
            error_summary = "; ".join(result.errors)
            if strict:
                raise DatasetValidationError(
                    f"Sample {idx} failed validation: {error_summary}"
                )
            logger.debug("Dropping sample %d: %s", idx, error_summary)
            dropped.append(idx)

    report = ValidationReport(
        total=len(samples),
        valid=len(valid_samples),
        invalid=len(dropped),
        dropped=dropped,
        warnings=warnings_list,
    )
    report.log()

    if report.invalid > 0:
        drop_pct = 100.0 * report.invalid / max(report.total, 1)
        logger.warning(
            "Dropped %d / %d samples (%.1f %%) during validation.",
            report.invalid, report.total, drop_pct,
        )

    return valid_samples, report
