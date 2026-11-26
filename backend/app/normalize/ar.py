"""Arabic orthographic normalization and optional morphology analysis."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

from .interface import LanguageNormalizer

LOGGER = logging.getLogger(__name__)
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})


def normalize_arabic(word: str) -> str:
    """Remove tashkeel/tatweel and unify common Arabic letter variants."""
    normalized = unicodedata.normalize("NFC", word).strip()
    normalized = "".join(
        character
        for character in normalized
        if character != "\u0640" and unicodedata.category(character) != "Mn"
    )
    return normalized.translate(_ALEF_VARIANTS).replace("ى", "ي")


@lru_cache(maxsize=1)
def _morphology_backend() -> tuple[str, object | None]:
    try:
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB

        return "camel", Analyzer(MorphologyDB.builtin_db())
    except (ImportError, OSError, RuntimeError, ValueError) as camel_error:
        try:
            from farasa.stemmer import FarasaStemmer

            binary_path = Path(FarasaStemmer.bin_dir) / "lib" / "FarasaSegmenterJar.jar"
            if not binary_path.is_file():
                raise FileNotFoundError(
                    "Farasa binaries are not installed; refusing an implicit download"
                )
            return "farasa", FarasaStemmer(interactive=True)
        except (ImportError, OSError, RuntimeError, ValueError) as farasa_error:
            LOGGER.warning(
                "Arabic morphology unavailable (CAMeL Tools: %s; Farasa: %s); "
                "using orthographic matching",
                camel_error,
                farasa_error,
            )
            return "none", None


def _optional_arabic_analysis(word: str) -> tuple[str, ...]:
    backend, analyzer = _morphology_backend()
    if backend == "camel" and analyzer is not None:
        analyses = analyzer.analyze(word)  # type: ignore[attr-defined]
        values = [
            analysis.get(field, "")
            for analysis in analyses
            for field in ("lex", "root")
            if analysis.get(field)
        ]
        return tuple(dict.fromkeys(normalize_arabic(value) for value in values))
    if backend == "farasa" and analyzer is not None:
        stem = analyzer.stem(word)  # type: ignore[attr-defined]
        return (normalize_arabic(stem),) if stem else ()
    return ()


class ArabicNormalizer(LanguageNormalizer):
    """Normalize Arabic script and expose optional lemma/root candidates."""

    language = "ar"

    def __init__(
        self,
        lemma_provider: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self._lemma_provider = lemma_provider or _optional_arabic_analysis

    def normalize_input(self, word: str) -> str:
        return normalize_arabic(word)

    def lookup_key(self, word: str) -> str:
        return normalize_arabic(word).replace("ة", "ه")

    def has_valid_characters(self, word: str) -> bool:
        return bool(word) and all(
            (
                unicodedata.name(character, "").startswith("ARABIC ")
                and (
                    unicodedata.category(character).startswith("L")
                    or unicodedata.category(character) == "Mn"
                )
            )
            for character in word
        )

    def lemma_candidates(self, word: str) -> tuple[str, ...]:
        return tuple(
            normalized
            for candidate in self._lemma_provider(word)
            if (normalized := normalize_arabic(candidate)) != word
        )
