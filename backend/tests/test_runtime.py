"""Tests for startup loading and in-memory asset validation."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app import runtime as runtime_module
from app.runtime import GameRuntime, InMemoryVectorStore

LANGUAGES = ("en", "de", "es", "ar", "ja")


def test_vector_store_reads_float16_matrix_into_process_memory(tmp_path: Path) -> None:
    binary_path = tmp_path / "vectors.f16.bin"
    index_path = tmp_path / "vectors.index.json"
    np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16).tofile(binary_path)
    index_path.write_text(
        json.dumps({"shape": [2, 2], "words": {"alpha": 0, "beta": 1}}),
        encoding="utf-8",
    )

    store = InMemoryVectorStore.from_files(binary_path, index_path)

    assert isinstance(store.matrix, np.ndarray)
    assert not isinstance(store.matrix, np.memmap)
    assert store.matrix.dtype == np.float16
    assert np.allclose(store.vector("alpha"), [1.0, 0.0])


def test_repository_runtime_loads_language_assets_and_puzzles_into_memory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data"
    (data_root / "vectors").mkdir(parents=True)
    (data_root / "puzzles").mkdir()
    vector_words = {"start": 0, "finish": 1}
    language_settings = {
        language: {"threshold": 0.5}
        for language in LANGUAGES
    }
    validator = SimpleNamespace(
        indexes={
            language: SimpleNamespace(terms=set(vector_words))
            for language in LANGUAGES
        }
    )
    def fake_from_repository(**kwargs):
        assert kwargs["data_root"] == data_root
        return validator

    monkeypatch.setattr(
        runtime_module,
        "NormalizationService",
        SimpleNamespace(from_repository=fake_from_repository),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_language_settings",
        lambda: language_settings,
    )

    for language in LANGUAGES:
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16).tofile(
            data_root / "vectors" / f"{language}.f16.bin"
        )
        (data_root / "vectors" / f"{language}.index.json").write_text(
            json.dumps({"shape": [2, 2], "words": vector_words}),
            encoding="utf-8",
        )
        (data_root / "puzzles" / f"{language}.json").write_text(
            json.dumps(
                {
                    "lang": language,
                    "daily_queue": [
                        {
                            "id": f"{language}-daily-2026-11-26",
                            "lang": language,
                            "anchorA": "start",
                            "anchorB": "finish",
                            "shortestPath": ["start", "finish"],
                            "alternativePaths": [],
                            "difficulty": "easy",
                            "date": "2026-11-26",
                        }
                    ],
                    "practice_pool": [
                        {
                            "id": f"{language}-practice-0001",
                            "lang": language,
                            "anchorA": "start",
                            "anchorB": "finish",
                            "shortestPath": ["start", "finish"],
                            "alternativePaths": [],
                            "difficulty": "easy",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    with sqlite3.connect(data_root / "linkwords.sqlite") as connection:
        connection.execute(
            """
            CREATE TABLE synonyms (
                lang TEXT, word_a TEXT, word_b TEXT, relation TEXT, boost_score REAL
            )
            """
        )
        connection.execute(
            "INSERT INTO synonyms VALUES ('en', 'finish', 'start', 'synonym', 0.9)"
        )

    runtime = GameRuntime.from_repository(
        data_root=data_root,
        repository_root=tmp_path,
    )

    assert set(runtime.vector_stores) == set(LANGUAGES)
    assert runtime.vector_stores["en"].matrix.dtype == np.float16
    assert runtime.lexical_scores["en"][("finish", "start")] == 0.9
    assert runtime.daily_puzzles["ja"][0].id == "ja-daily-2026-11-26"
