"""Public result types for multilingual word validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ValidationReason(StrEnum):
    """Reason a candidate word failed normalization or validation."""

    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NOT_IN_VOCAB = "not_in_vocab"
    BLOCKED = "blocked"
    INVALID_CHARS = "invalid_chars"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Normalized value, user-facing spelling, lemma, and validation status."""

    ok: bool
    normalized: str
    display: str
    lemma: str | None
    reason: ValidationReason | None
