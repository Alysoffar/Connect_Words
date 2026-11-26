"""Startup-built vocabulary and orthographic lookup indexes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from .interface import LanguageNormalizer


class VocabularyIndex:
    """Index canonical vocabulary terms by exact form and lookup equivalence."""

    def __init__(
        self,
        vocabulary: Iterable[str],
        normalizer: LanguageNormalizer,
        *,
        aliases: Mapping[str, set[str]] | None = None,
    ) -> None:
        """Build exact-term, orthographic, and explicit alias maps."""
        self.normalizer = normalizer
        self.terms = {
            normalized
            for word in vocabulary
            if (normalized := normalizer.normalize_input(word))
        }
        by_key: dict[str, set[str]] = defaultdict(set)
        for term in self.terms:
            by_key[normalizer.lookup_key(term)].add(term)
        self.by_key = dict(by_key)
        self.aliases = {
            key: filtered
            for key, values in (aliases or {}).items()
            if (filtered := set(values) & self.terms)
        }

    def resolve(self, candidates: Iterable[str]) -> str | None:
        """Resolve in order: exact lemma, explicit alias, then orthographic key."""
        normalized_candidates = [
            self.normalizer.normalize_input(candidate)
            for candidate in candidates
            if candidate
        ]
        for candidate in normalized_candidates:
            matches = self.aliases.get(self.normalizer.lookup_key(candidate), set())
            if len(matches) == 1:
                return next(iter(matches))

        for candidate in normalized_candidates:
            if candidate in self.terms:
                return candidate

        for candidate in normalized_candidates:
            matches = self.by_key.get(self.normalizer.lookup_key(candidate), set())
            if matches:
                return sorted(matches)[0]
        return None

    def contains_blocked(
        self,
        candidates: Iterable[str],
        blocked_keys: set[str],
    ) -> bool:
        """Check candidate forms against pre-normalized blocklist keys."""
        for candidate in candidates:
            key = self.normalizer.lookup_key(candidate)
            if key in blocked_keys:
                return True
            if any(
                self.normalizer.lookup_key(alias) in blocked_keys
                for alias in self.aliases.get(key, set())
            ):
                return True
        return False
