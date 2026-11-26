"""English normalization, punctuation handling, and inflection lemmatization."""

from __future__ import annotations

import logging
import string
import unicodedata
from functools import lru_cache

from .dependencies import repository_nltk_path
from .interface import LanguageNormalizer

LOGGER = logging.getLogger(__name__)
_IRREGULARS = {
    "ate": ("eat",),
    "better": ("good",),
    "children": ("child",),
    "did": ("do",),
    "does": ("do",),
    "feet": ("foot",),
    "geese": ("goose",),
    "men": ("man",),
    "mice": ("mouse",),
    "ran": ("run",),
    "went": ("go",),
    "worse": ("bad",),
    "women": ("woman",),
}
_ASCII_PUNCTUATION = set(string.punctuation)


def _fallback_lemmas(word: str) -> tuple[str, ...]:
    candidates: list[str] = list(_IRREGULARS.get(word, ()))
    if len(word) > 4 and word.endswith("ies"):
        candidates.append(word[:-3] + "y")
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        candidates.extend((stem, stem + "e"))
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        candidates.extend((stem, stem + "e"))
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
    if len(word) > 4 and word.endswith("es"):
        candidates.extend((word[:-2], word[:-1]))
    if len(word) > 3 and word.endswith("s"):
        candidates.append(word[:-1])
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate != word))


@lru_cache(maxsize=1)
def _wordnet_lemmatizer():
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        nltk_path = str(repository_nltk_path())
        if nltk_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_path)
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests", pos="n")
        return lemmatizer
    except (ImportError, LookupError) as error:
        LOGGER.info(
            "NLTK WordNet lemmatizer unavailable; using English fallback rules: %s",
            error,
        )
        return None


class EnglishNormalizer(LanguageNormalizer):
    """Normalize English words to lowercase and derive WordNet lemmas."""

    language = "en"

    def normalize_input(self, word: str) -> str:
        normalized = unicodedata.normalize("NFKC", word).casefold().strip()
        return "".join(
            character
            for character in normalized
            if character not in _ASCII_PUNCTUATION
            and not unicodedata.category(character).startswith("P")
        )

    def lookup_key(self, word: str) -> str:
        return word.casefold()

    def has_valid_characters(self, word: str) -> bool:
        return bool(word) and all(character.isalpha() for character in word)

    def lemma_candidates(self, word: str) -> tuple[str, ...]:
        lemmas: list[str] = []
        lemmatizer = _wordnet_lemmatizer()
        if lemmatizer is not None:
            for part_of_speech in ("n", "v", "a", "r"):
                lemma = lemmatizer.lemmatize(word, pos=part_of_speech)
                if lemma != word and lemma not in lemmas:
                    lemmas.append(lemma)
        lemmas.extend(
            candidate
            for candidate in _fallback_lemmas(word)
            if candidate not in lemmas
        )
        return tuple(lemmas)
