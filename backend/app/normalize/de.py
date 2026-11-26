"""German normalization and ß/ss orthographic lookup equivalence."""

from __future__ import annotations

import unicodedata

from .interface import LanguageNormalizer


class GermanNormalizer(LanguageNormalizer):
    """Preserve display spelling while resolving case and ß/ss variants."""

    language = "de"

    def normalize_input(self, word: str) -> str:
        return unicodedata.normalize("NFC", word).lower().strip()

    def lookup_key(self, word: str) -> str:
        return self.normalize_input(word).casefold().replace("ß", "ss")

    def has_valid_characters(self, word: str) -> bool:
        return bool(word) and all(
            character.isalpha()
            or unicodedata.category(character).startswith("M")
            for character in word
        )

    def lemma_candidates(self, _word: str) -> tuple[str, ...]:
        del _word
        return ()
