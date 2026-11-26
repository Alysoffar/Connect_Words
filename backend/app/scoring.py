"""Fast in-memory equivalent of the pipeline's semantic score function."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .graph import canonical_edge_key


class VectorTable(Protocol):
    """Vector rows keyed by normalized vocabulary terms."""

    matrix: NDArray[np.float16]
    words: Mapping[str, int]

    def vector(self, word: str) -> NDArray[np.float32]: ...


LexicalScores = Mapping[tuple[str, str], float]


def score(
    language: str,
    word_a: str,
    word_b: str,
    *,
    vector_store: VectorTable,
    lexical_scores: LexicalScores,
) -> float:
    """Return max(cosine similarity, WordNet boost) from startup-loaded assets."""
    del language
    left = vector_store.vector(word_a)
    right = vector_store.vector(word_b)
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    lexical = lexical_scores.get(canonical_edge_key(word_a, word_b), 0.0)
    return max(cosine, lexical)


def score_existing_pairs(
    words: Sequence[str],
    *,
    vector_store: VectorTable,
    lexical_scores: LexicalScores,
) -> dict[tuple[str, str], float]:
    """Score the existing board in one matrix operation for stateless requests."""
    rows = [vector_store.words[word] for word in words]
    vectors = np.asarray(vector_store.matrix[rows], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0) or not np.isfinite(vectors).all():
        raise ValueError("Board vectors must be finite and nonzero")
    vectors /= norms[:, None]
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)

    result: dict[tuple[str, str], float] = {}
    for row, word_a in enumerate(words):
        for column in range(row + 1, len(words)):
            word_b = words[column]
            key = canonical_edge_key(word_a, word_b)
            result[key] = max(
                float(similarities[row, column]),
                lexical_scores.get(key, 0.0),
            )
    return result
