"""Shared startup index, blocklist loading, and validation orchestration."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from . import ar, de, en, es, ja
from .dependencies import REPOSITORY_ROOT, load_language_settings
from .index import VocabularyIndex
from .interface import LanguageNormalizer
from .models import ValidationReason, ValidationResult

LOGGER = logging.getLogger(__name__)
NORMALIZERS: dict[str, LanguageNormalizer] = {
    "en": en.EnglishNormalizer(),
    "de": de.GermanNormalizer(),
    "ar": ar.ArabicNormalizer(),
    "es": es.SpanishNormalizer(),
    "ja": ja.JapaneseNormalizer(),
}


class NormalizationService:
    """Own per-language vocab indexes and blocklists for one backend process."""

    def __init__(
        self,
        vocabularies: Mapping[str, Iterable[str]],
        *,
        blocklists: Mapping[str, Iterable[str]] | None = None,
        normalizers: Mapping[str, LanguageNormalizer] = NORMALIZERS,
    ) -> None:
        """Build all orthographic indexes once when the service starts."""
        self.settings = load_language_settings()
        supported = set(self.settings)
        if set(vocabularies) != supported:
            raise ValueError(
                "Vocabulary data must be provided for every configured language: "
                + ", ".join(sorted(supported))
            )
        blocklists = blocklists or {}
        self.indexes: dict[str, VocabularyIndex] = {}
        self.blocked_keys: dict[str, set[str]] = {}
        self.normalizers = dict(normalizers)
        for language in self.settings:
            normalizer = self.normalizers.get(language)
            if normalizer is None:
                raise ValueError(f"No normalizer registered for {language}")
            terms = {
                normalized
                for word in vocabularies[language]
                if (normalized := normalizer.normalize_input(word))
            }
            aliases = normalizer.build_aliases(terms)
            self.indexes[language] = VocabularyIndex(
                terms,
                normalizer,
                aliases=aliases,
            )
            self.blocked_keys[language] = {
                normalizer.lookup_key(normalizer.normalize_input(word))
                for word in blocklists.get(language, ())
                if normalizer.normalize_input(word)
            }
            LOGGER.info(
                "Built %s vocabulary index: %d terms, %d orthographic keys, %d aliases, "
                "%d blocked terms",
                language,
                len(terms),
                len(self.indexes[language].by_key),
                len(aliases),
                len(self.blocked_keys[language]),
            )

    @classmethod
    def from_repository(
        cls,
        *,
        data_root: Path | None = None,
        blocklist_root: Path | None = None,
    ) -> "NormalizationService":
        """Load each language vocabulary and shared blocklist from repository data."""
        data_root = data_root or REPOSITORY_ROOT / "data"
        blocklist_root = blocklist_root or REPOSITORY_ROOT / "shared" / "blocklists"
        settings = load_language_settings()
        vocabularies: dict[str, list[str]] = {}
        blocklists: dict[str, list[str]] = {}
        for language in settings:
            vocabulary_path = data_root / "vocab" / f"{language}.json"
            if not vocabulary_path.is_file():
                raise FileNotFoundError(f"Missing vocabulary file: {vocabulary_path}")
            raw_vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
            if not isinstance(raw_vocabulary, list) or not all(
                isinstance(word, str) for word in raw_vocabulary
            ):
                raise ValueError(f"Invalid vocabulary JSON: {vocabulary_path}")
            vocabularies[language] = raw_vocabulary

            blocklist_path = blocklist_root / f"{language}.txt"
            if not blocklist_path.is_file():
                raise FileNotFoundError(f"Missing blocklist file: {blocklist_path}")
            blocklists[language] = [
                line.strip()
                for line in blocklist_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        return cls(vocabularies, blocklists=blocklists)

    def normalize_and_validate(self, word: str, lang: str) -> ValidationResult:
        """Normalize a user word and validate syntax, length, blocks, and vocabulary."""
        if lang not in self.settings:
            raise ValueError(f"Unsupported language: {lang}")
        if not isinstance(word, str):
            return ValidationResult(
                False, "", "", None, ValidationReason.INVALID_CHARS
            )

        normalizer = self.normalizers[lang]
        display = word.strip()
        normalized = normalizer.normalize_input(display)
        if not normalizer.has_valid_characters(normalized):
            return ValidationResult(
                False, normalized, display, None, ValidationReason.INVALID_CHARS
            )

        length = len(normalized)
        settings = self.settings[lang]
        if (
            length < settings["min_word_length"] or length == 1
        ) and not (
            length == 1 and normalizer.permits_single_character(normalized)
        ):
            return ValidationResult(
                False, normalized, display, None, ValidationReason.TOO_SHORT
            )
        if length > settings["max_word_length"]:
            return ValidationResult(
                False, normalized, display, None, ValidationReason.TOO_LONG
            )

        index = self.indexes[lang]
        if index.contains_blocked((normalized,), self.blocked_keys[lang]):
            return ValidationResult(
                False, normalized, display, None, ValidationReason.BLOCKED
            )

        candidates = tuple(
            dict.fromkeys((*normalizer.lemma_candidates(normalized), normalized))
        )
        if index.contains_blocked(candidates, self.blocked_keys[lang]):
            return ValidationResult(
                False, normalized, display, None, ValidationReason.BLOCKED
            )

        resolved = index.resolve(candidates)
        if resolved is None:
            return ValidationResult(
                False, normalized, display, None, ValidationReason.NOT_IN_VOCAB
            )
        return ValidationResult(True, resolved, display, resolved, None)


_default_service: NormalizationService | None = None


def configure_normalizer(service: NormalizationService) -> None:
    """Set the process-wide service during application startup."""
    global _default_service
    _default_service = service


def normalize_and_validate(word: str, lang: str) -> ValidationResult:
    """Validate a word using the configured service or repository startup data."""
    global _default_service
    if _default_service is None:
        _default_service = NormalizationService.from_repository()
    return _default_service.normalize_and_validate(word, lang)
