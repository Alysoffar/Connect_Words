"""Multilingual word normalization and vocabulary validation."""

from .models import ValidationReason, ValidationResult
from .service import (
    NormalizationService,
    configure_normalizer,
    normalize_and_validate,
)

__all__ = [
    "NormalizationService",
    "ValidationReason",
    "ValidationResult",
    "configure_normalizer",
    "normalize_and_validate",
]
