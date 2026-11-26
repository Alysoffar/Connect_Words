"""Tests for lexical boost storage and combined word-pair scoring."""

import sqlite3
from pathlib import Path

import numpy as np

from linkwords_pipeline.lexical import (
    create_lexical_schema,
    lexical_boost_score,
)
from linkwords_pipeline.scoring import score
from linkwords_pipeline.vectors import VectorStore


def _write_vector_store(directory: Path) -> VectorStore:
    matrix = np.zeros((2, 4), dtype=np.float16)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    binary_path = directory / "vectors.f16.bin"
    index_path = directory / "vectors.index.json"
    matrix.tofile(binary_path)
    index_path.write_text(
        '{"shape":[2,4],"dtype":"float16","words":{"dog":0,"animal":1}}',
        encoding="utf-8",
    )
    return VectorStore(binary_path, index_path)


def _write_lexical_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        create_lexical_schema(connection)
        connection.execute(
            "INSERT INTO synonyms VALUES (?, ?, ?, ?, ?)",
            ("en", "animal", "dog", "hypernym", 0.75),
        )


def test_known_taxonomic_pair_scores_at_least_point_seven_five(tmp_path: Path) -> None:
    database_path = tmp_path / "linkwords.sqlite"
    _write_lexical_db(database_path)

    assert lexical_boost_score("en", "dog", "animal", database_path) >= 0.75
    assert score(
        "en",
        "dog",
        "animal",
        vector_store=_write_vector_store(tmp_path),
        database_path=database_path,
    ) >= 0.75


def test_lexical_score_is_symmetric_and_uses_highest_relation(tmp_path: Path) -> None:
    database_path = tmp_path / "linkwords.sqlite"
    _write_lexical_db(database_path)

    assert lexical_boost_score("en", "animal", "dog", database_path) == 0.75
    assert lexical_boost_score("en", "dog", "unknown", database_path) == 0.0
