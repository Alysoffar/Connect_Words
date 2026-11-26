"""Unicode-aware word normalization shared by pipeline validation steps."""

from __future__ import annotations

import unicodedata

WORDNET_LANGUAGE_CODES = {
    "en": "eng",
    "de": "deu",
    "es": "spa",
    "ar": "arb",
    "ja": "jpn",
}


def normalize_word(word: str, language: str) -> str:
    """Normalize a candidate word consistently for its language."""
    if language == "ja":
        return unicodedata.normalize("NFKC", word).strip()

    normalized = unicodedata.normalize("NFC", word).strip().casefold()
    if language == "en":
        return "".join(
            character
            for character in unicodedata.normalize("NFKC", word).casefold().strip()
            if not unicodedata.category(character).startswith("P")
        )
    if language == "ar":
        normalized = "".join(
            character
            for character in normalized
            if character != "\u0640" and unicodedata.category(character) != "Mn"
        )
        normalized = normalized.translate(
            str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})
        ).replace("ى", "ي")
    return unicodedata.normalize("NFC", normalized)


def is_single_kanji(word: str) -> bool:
    """Return whether a one-character string is a CJK unified ideograph."""
    return (
        len(word) == 1
        and unicodedata.name(word, "").startswith("CJK UNIFIED IDEOGRAPH")
    )


def allows_single_character_word(word: str, language: str) -> bool:
    """Apply the supported language's exception for valid single-character words."""
    return language == "ja" and is_single_kanji(word)
