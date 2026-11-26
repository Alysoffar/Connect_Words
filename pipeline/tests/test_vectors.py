"""Tests for hybrid vector construction and normalization."""

import numpy as np
import pytest

from linkwords_pipeline.vectors import (
    BGE_DIMENSION,
    NUMBERBATCH_DIMENSION,
    _cache_bge,
    build_hybrid_matrix,
    l2_normalize_rows,
)


def test_hybrid_matrix_has_expected_shape_and_unit_rows() -> None:
    numberbatch = np.zeros(NUMBERBATCH_DIMENSION, dtype=np.float32)
    numberbatch[0] = 1.0
    bge_a = np.zeros(BGE_DIMENSION, dtype=np.float32)
    bge_a[0] = 1.0
    bge_b = np.zeros(BGE_DIMENSION, dtype=np.float32)
    bge_b[1] = 1.0

    matrix = build_hybrid_matrix(
        ["alpha", "beta"],
        {"alpha": numberbatch, "beta": numberbatch},
        {"alpha": bge_a, "beta": bge_b},
        {"numberbatch": 0.25, "bge_m3": 0.75},
    )

    assert matrix.shape == (2, NUMBERBATCH_DIMENSION + BGE_DIMENSION)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.dot(matrix[0], matrix[1]), 0.25, atol=1e-6)


def test_hybrid_matrix_renormalizes_weights_for_available_sources() -> None:
    first = np.zeros(BGE_DIMENSION, dtype=np.float32)
    first[0] = 2.0
    second = np.zeros(BGE_DIMENSION, dtype=np.float32)
    second[0] = 5.0

    matrix = build_hybrid_matrix(
        ["alpha", "beta"],
        {},
        {"alpha": first, "beta": second},
        {"numberbatch": 0.25, "bge_m3": 0.75},
    )

    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.dot(matrix[0], matrix[1]), 1.0, atol=1e-6)


def test_source_vectors_are_normalized_before_blending() -> None:
    numberbatch_a = np.zeros(NUMBERBATCH_DIMENSION, dtype=np.float32)
    numberbatch_a[0] = 100.0
    numberbatch_b = np.zeros(NUMBERBATCH_DIMENSION, dtype=np.float32)
    numberbatch_b[0] = 1.0
    bge = np.zeros(BGE_DIMENSION, dtype=np.float32)
    bge[0] = 1.0

    matrix = build_hybrid_matrix(
        ["alpha", "beta"],
        {"alpha": numberbatch_a, "beta": numberbatch_b},
        {"alpha": bge, "beta": bge},
        {"numberbatch": 0.25, "bge_m3": 0.75},
    )

    np.testing.assert_allclose(np.dot(matrix[0], matrix[1]), 1.0, atol=1e-6)


def test_normalization_rejects_bad_shapes_and_zero_rows() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        l2_normalize_rows(np.ones(3))
    with pytest.raises(ValueError, match="zero-length"):
        l2_normalize_rows(np.zeros((1, 3)))


def test_hybrid_matrix_requires_at_least_one_source_per_word() -> None:
    with pytest.raises(ValueError, match="No vector source"):
        build_hybrid_matrix(
            ["missing"],
            {},
            {},
            {"numberbatch": 0.25, "bge_m3": 0.75},
        )


def test_bge_cache_reuses_existing_words_and_encodes_only_new_terms(
    tmp_path,
    monkeypatch,
) -> None:
    import linkwords_pipeline.vectors as vectors

    monkeypatch.setattr(vectors, "CACHE_ROOT", tmp_path)
    calls: list[list[str]] = []

    def fake_encode(words, _model_path, *, device, batch_size):
        del device, batch_size
        calls.append(list(words))
        return {
            word: np.eye(1, BGE_DIMENSION, index, dtype=np.float32)[0]
            for index, word in enumerate(words)
        }

    monkeypatch.setattr(vectors, "_encode_bge", fake_encode)
    _cache_bge(
        "ja",
        ["old", "kept"],
        "first-vocabulary",
        model_path=tmp_path,
        device="cpu",
        batch_size=1,
    )
    result = _cache_bge(
        "ja",
        ["kept", "new"],
        "second-vocabulary",
        model_path=tmp_path,
        device="cpu",
        batch_size=1,
    )

    assert calls == [["old", "kept"], ["new"]]
    assert set(result) == {"kept", "new"}
