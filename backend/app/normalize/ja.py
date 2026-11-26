"""Japanese normalization, morphological lemmas, and reading aliases."""

from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from functools import lru_cache

from .interface import LanguageNormalizer

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _sudachi_tokenizer():
    try:
        from sudachipy import Dictionary

        return Dictionary().create()
    except (ImportError, OSError, RuntimeError):
        return None


@lru_cache(maxsize=1)
def _fugashi_tagger():
    try:
        import fugashi

        return fugashi.Tagger()
    except (ImportError, OSError, RuntimeError):
        return None


def _sudachi_forms(word: str) -> tuple[str, ...]:
    tokenizer = _sudachi_tokenizer()
    if tokenizer is None:
        return ()
    from sudachipy import Tokenizer

    tokens = list(tokenizer.tokenize(word, Tokenizer.SplitMode.C))
    if len(tokens) != 1:
        return ()
    token = tokens[0]
    return tuple(
        dict.fromkeys(
            value
            for value in (token.dictionary_form(), token.reading_form())
            if value and value != "*"
        )
    )


def _fugashi_forms(word: str) -> tuple[str, ...]:
    tagger = _fugashi_tagger()
    if tagger is None:
        return ()
    tokens = list(tagger(word))
    if len(tokens) != 1:
        return ()
    feature = tokens[0].feature
    base = getattr(feature, "orthBase", None) or getattr(feature, "lemma", None)
    reading = getattr(feature, "kana", None) or getattr(feature, "pron", None)
    return tuple(value for value in (base, reading) if value and value != "*")


class JapaneseNormalizer(LanguageNormalizer):
    """Apply NFKC, tokenizer lemmas, and unique Kana-to-Kanji aliases."""

    language = "ja"

    def normalize_input(self, word: str) -> str:
        return unicodedata.normalize("NFKC", word).strip()

    def lookup_key(self, word: str) -> str:
        return self.normalize_input(word)

    def has_valid_characters(self, word: str) -> bool:
        return bool(word) and all(
            unicodedata.category(character).startswith("L")
            or unicodedata.category(character).startswith("M")
            for character in word
        )

    def permits_single_character(self, word: str) -> bool:
        """Allow single CJK ideographs, but not single Kana or Latin letters."""
        return len(word) == 1 and unicodedata.name(word, "").startswith(
            ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
        )

    def lemma_candidates(self, word: str) -> tuple[str, ...]:
        forms = _sudachi_forms(word)
        if not forms:
            forms = _fugashi_forms(word)
        return tuple(
            dict.fromkeys(
                self.normalize_input(form)
                for form in forms
                if self.normalize_input(form) != word
            )
        )

    def build_aliases(self, vocabulary: set[str]) -> dict[str, set[str]]:
        if _sudachi_tokenizer() is None and _fugashi_tagger() is None:
            LOGGER.warning(
                "Japanese morphology unavailable; Kana reading aliases will not be built"
            )
        reading_to_lemmas: dict[str, set[str]] = defaultdict(set)
        for term in vocabulary:
            forms = _sudachi_forms(term)
            if not forms:
                forms = _fugashi_forms(term)
            if len(forms) >= 2:
                base, reading = forms[0], forms[-1]
                canonical_base = self.normalize_input(base)
                canonical_reading = self.lookup_key(reading)
                if canonical_base in vocabulary and canonical_reading != canonical_base:
                    reading_to_lemmas[canonical_reading].add(canonical_base)
        return {
            reading: lemmas
            for reading, lemmas in reading_to_lemmas.items()
            if len(lemmas) == 1
        }
