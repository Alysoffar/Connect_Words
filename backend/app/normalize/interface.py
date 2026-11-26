"""Shared contract implemented by each language-specific normalizer."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LanguageNormalizer(ABC):
    """Language hooks used by the shared validation service and vocabulary index."""

    language: str

    @abstractmethod
    def normalize_input(self, word: str) -> str:
        """Return the language's canonical form before vocabulary lookup."""

    @abstractmethod
    def lookup_key(self, word: str) -> str:
        """Return a key that applies this language's lookup equivalences."""

    @abstractmethod
    def has_valid_characters(self, word: str) -> bool:
        """Return whether a normalized word contains only permitted characters."""

    @abstractmethod
    def lemma_candidates(self, word: str) -> tuple[str, ...]:
        """Return lemma/root candidates, excluding the unmodified word."""

    def permits_single_character(self, word: str) -> bool:
        """Return whether this language allows a one-character vocabulary word."""
        return False

    def build_aliases(self, vocabulary: set[str]) -> dict[str, set[str]]:
        """Build startup aliases such as Japanese readings to vocabulary lemmas."""
        del vocabulary
        return {}
