"""Spanish normalization and accent-insensitive vocabulary lookup."""

from __future__ import annotations

import unicodedata

from .interface import LanguageNormalizer


def accent_insensitive_key(word: str) -> str:
    """Remove accents but preserve ñ as a distinct Spanish letter."""
    decomposed = unicodedata.normalize("NFD", word.casefold())
    output: list[str] = []
    for index, character in enumerate(decomposed):
        if unicodedata.category(character) != "Mn":
            output.append(character)
        elif character == "\u0303" and index > 0 and decomposed[index - 1] == "n":
            output.append(character)
    return "".join(output)


class SpanishNormalizer(LanguageNormalizer):
    """Normalize Spanish spelling and use an accent-tolerant lookup key."""

    language = "es"

    def normalize_input(self, word: str) -> str:
        return unicodedata.normalize("NFC", word).casefold().strip()

    def lookup_key(self, word: str) -> str:
        return accent_insensitive_key(word)

    def has_valid_characters(self, word: str) -> bool:
        return bool(word) and all(
            character.isalpha()
            or unicodedata.category(character).startswith("M")
            for character in word
        )

    def lemma_candidates(self, _word: str) -> tuple[str, ...]:
        del _word
        return ()
