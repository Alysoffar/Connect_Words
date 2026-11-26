"""Combine hybrid cosine similarity with direct WordNet boosts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import LANGUAGE_CODES, REPOSITORY_ROOT
from .lexical import DATABASE_PATH, lexical_boost_score
from .normalization import normalize_word
from .vectors import VectorStore, cosine_similarity


@lru_cache(maxsize=len(LANGUAGE_CODES))
def _load_vector_store(language: str) -> VectorStore:
    vector_root = REPOSITORY_ROOT / "data" / "vectors"
    return VectorStore(
        vector_root / f"{language}.f16.bin",
        vector_root / f"{language}.index.json",
    )


def score(
    language: str,
    word_a: str,
    word_b: str,
    *,
    vector_store: VectorStore | None = None,
    database_path: Path = DATABASE_PATH,
) -> float:
    """Score a word pair as max(blended cosine similarity, lexical boost)."""
    if language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language: {language}")
    normalized_a = normalize_word(word_a, language)
    normalized_b = normalize_word(word_b, language)
    store = vector_store or _load_vector_store(language)
    cosine_blend = cosine_similarity(store, normalized_a, normalized_b)
    boost_score = lexical_boost_score(
        language,
        normalized_a,
        normalized_b,
        database_path,
    )
    return max(cosine_blend, boost_score)
