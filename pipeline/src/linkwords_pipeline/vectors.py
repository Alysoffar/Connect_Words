"""Build and load normalized hybrid semantic vector stores."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .config import REPOSITORY_ROOT, get_language_settings
from .normalization import normalize_word

NUMBERBATCH_URL = (
    "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
    "numberbatch-19.08.txt.gz"
)
NUMBERBATCH_DIMENSION = 300
BGE_DIMENSION = 1024
BGE_MODEL_ID = "BAAI/bge-m3"
BGE_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_COUNT = 60_000
VECTOR_ROOT = REPOSITORY_ROOT / "data" / "vectors"
MODEL_ROOT = REPOSITORY_ROOT / "data" / "models" / "bge-m3"
CACHE_ROOT = REPOSITORY_ROOT / "data" / "cache" / "vectors"

FloatMatrix = NDArray[np.float32]


def l2_normalize_rows(vectors: NDArray[np.floating[Any]]) -> FloatMatrix:
    """Return finite, nonzero rows normalized to unit L2 length."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Vector data must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Vector data contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Vector data contains a zero-length row")
    return matrix / norms


def build_hybrid_matrix(
    words: Sequence[str],
    numberbatch_vectors: Mapping[str, NDArray[np.floating[Any]]],
    bge_vectors: Mapping[str, NDArray[np.floating[Any]]],
    weights: Mapping[str, float],
) -> FloatMatrix:
    """Combine normalized source vectors with configured cosine blend weights."""
    if set(weights) != {"numberbatch", "bge_m3"}:
        raise ValueError("weights must contain Numberbatch and BGE-M3")
    weight_total = sum(weights.values())
    if weight_total <= 0 or any(weight < 0 for weight in weights.values()):
        raise ValueError("vector weights must be non-negative with positive sum")

    result = np.zeros((len(words), NUMBERBATCH_DIMENSION + BGE_DIMENSION), dtype=np.float32)
    for row, word in enumerate(words):
        numberbatch = numberbatch_vectors.get(word)
        bge = bge_vectors.get(word)
        available_weight = 0.0
        if numberbatch is not None:
            vector = np.asarray(numberbatch, dtype=np.float32)
            if vector.shape != (NUMBERBATCH_DIMENSION,):
                raise ValueError(f"Numberbatch vector for {word!r} has an invalid shape")
            result[row, :NUMBERBATCH_DIMENSION] = l2_normalize_rows(vector[None, :])[0]
            available_weight += weights["numberbatch"]
        if bge is not None:
            vector = np.asarray(bge, dtype=np.float32)
            if vector.shape != (BGE_DIMENSION,):
                raise ValueError(f"BGE-M3 vector for {word!r} has an invalid shape")
            result[row, NUMBERBATCH_DIMENSION:] = l2_normalize_rows(vector[None, :])[0]
            available_weight += weights["bge_m3"]
        if available_weight == 0:
            raise ValueError(f"No vector source contains {word!r}")

        numberbatch_scale = math.sqrt(weights["numberbatch"] / available_weight)
        bge_scale = math.sqrt(weights["bge_m3"] / available_weight)
        result[row, :NUMBERBATCH_DIMENSION] *= numberbatch_scale
        result[row, NUMBERBATCH_DIMENSION:] *= bge_scale

    return l2_normalize_rows(result)


def _source_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _read_numberbatch(
    path: Path,
    language: str,
    vocabulary: set[str],
) -> dict[str, NDArray[np.float32]]:
    vectors: dict[str, NDArray[np.float32]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as source_file:
        for line in tqdm(source_file, desc=f"Reading Numberbatch ({language})"):
            fields = line.rstrip().split(" ")
            if len(fields) == 2 and fields[0].isdigit():
                continue
            if len(fields) != NUMBERBATCH_DIMENSION + 1:
                continue
            uri, *values = fields
            parts = uri.split("/")
            if len(parts) < 4 or parts[2] != language:
                continue
            word = normalize_word(
                urllib.parse.unquote("/".join(parts[3:]).replace("_", " ")),
                language,
            )
            if word in vocabulary:
                vectors[word] = np.asarray(values, dtype=np.float32)
    return vectors


def _encode_bge(
    words: Sequence[str],
    model_path: Path,
    *,
    device: str | None,
    batch_size: int,
) -> dict[str, NDArray[np.float32]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=device)
    encoded = model.encode(
        list(words),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    matrix = np.asarray(encoded, dtype=np.float32)
    if matrix.shape != (len(words), BGE_DIMENSION):
        raise ValueError(
            f"BGE-M3 produced shape {matrix.shape}; "
            f"expected ({len(words)}, {BGE_DIMENSION})"
        )
    matrix = l2_normalize_rows(matrix)
    return dict(zip(words, matrix, strict=True))


def _cache_bge(
    language: str,
    words: Sequence[str],
    vocabulary_hash: str,
    *,
    model_path: Path,
    device: str | None,
    batch_size: int,
) -> dict[str, NDArray[np.float32]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    matrix_path = CACHE_ROOT / f"{language}.bge.f16.bin"
    metadata_path = matrix_path.with_suffix(".json")
    expected = {
        "vocabulary_sha256": vocabulary_hash,
        "model_id": BGE_MODEL_ID,
        "model_revision": BGE_MODEL_REVISION,
        "dimension": BGE_DIMENSION,
        "words": list(words),
    }

    cached_vectors: dict[str, NDArray[np.float32]] = {}
    if matrix_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_words = metadata.get("words")
        if (
            metadata.get("model_id") == BGE_MODEL_ID
            and metadata.get("model_revision") == BGE_MODEL_REVISION
            and metadata.get("dimension") == BGE_DIMENSION
            and isinstance(cached_words, list)
            and all(isinstance(word, str) for word in cached_words)
        ):
            matrix = np.fromfile(matrix_path, dtype=np.float16)
            if matrix.size == len(cached_words) * BGE_DIMENSION:
                matrix = matrix.reshape((len(cached_words), BGE_DIMENSION)).astype(
                    np.float32
                )
                cached_vectors = dict(
                    zip(cached_words, l2_normalize_rows(matrix), strict=True)
                )

    if all(word in cached_vectors for word in words):
        return {word: cached_vectors[word] for word in words}

    missing_words = [word for word in words if word not in cached_vectors]
    encoded = _encode_bge(
        missing_words,
        model_path,
        device=device,
        batch_size=batch_size,
    )
    combined = {
        word: cached_vectors[word] if word in cached_vectors else encoded[word]
        for word in words
    }
    matrix = np.stack([combined[word] for word in words]).astype(np.float16)
    _atomic_write_bytes(matrix_path, matrix.tobytes())
    _atomic_write_bytes(
        metadata_path,
        (json.dumps(expected, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return combined


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def build_vector_store(
    language: str,
    *,
    vocabulary_path: Path | None = None,
    numberbatch_path: Path | None = None,
    model_path: Path = MODEL_ROOT,
    output_root: Path = VECTOR_ROOT,
    device: str | None = None,
    batch_size: int = 64,
    force: bool = False,
) -> Path:
    """Build a float16 hybrid matrix and its word-to-row index."""
    vocabulary_file = vocabulary_path or (
        REPOSITORY_ROOT / "data" / "vocab" / f"{language}.json"
    )
    numberbatch_file = numberbatch_path or (
        REPOSITORY_ROOT / "data" / "sources" / "numberbatch-19.08.txt.gz"
    )
    vocabulary: list[str] = json.loads(vocabulary_file.read_text(encoding="utf-8"))
    vocabulary_hash = hashlib.sha256(
        json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    output_root.mkdir(parents=True, exist_ok=True)
    binary_path = output_root / f"{language}.f16.bin"
    index_path = output_root / f"{language}.index.json"
    manifest_path = output_root / f"{language}.manifest.json"
    metadata = {
        "language": language,
        "vocabulary_sha256": vocabulary_hash,
        "numberbatch": _source_signature(numberbatch_file),
        "model_id": BGE_MODEL_ID,
        "model_revision": BGE_MODEL_REVISION,
        "shape": [len(vocabulary), NUMBERBATCH_DIMENSION + BGE_DIMENSION],
        "dtype": "float16",
    }
    expected_size = len(vocabulary) * (NUMBERBATCH_DIMENSION + BGE_DIMENSION) * 2
    if not force and all(path.is_file() for path in (binary_path, index_path, manifest_path)):
        if (
            json.loads(manifest_path.read_text(encoding="utf-8")) == metadata
            and binary_path.stat().st_size == expected_size
        ):
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("shape") == metadata["shape"] and len(index.get("words", {})) == len(vocabulary):
                return binary_path

    settings = get_language_settings(language)
    numberbatch = _read_numberbatch(numberbatch_file, language, set(vocabulary))
    bge_words = vocabulary[:EMBEDDING_COUNT]
    bge = _cache_bge(
        language,
        bge_words,
        vocabulary_hash,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )
    matrix = build_hybrid_matrix(
        vocabulary,
        numberbatch,
        bge,
        settings["cosine_blend_weights"],
    )

    temporary_binary = binary_path.with_suffix(".f16.bin.tmp")
    matrix.astype(np.float16).tofile(temporary_binary)
    os.replace(temporary_binary, binary_path)
    index = {
        "shape": list(matrix.shape),
        "dtype": "float16",
        "words": {word: row for row, word in enumerate(vocabulary)},
    }
    _atomic_write_bytes(
        index_path,
        (json.dumps(index, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    _atomic_write_bytes(
        manifest_path,
        (json.dumps(metadata, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return binary_path


class VectorStore:
    """Read-only memory-mapped vector data and its vocabulary index."""

    def __init__(self, binary_path: Path, index_path: Path) -> None:
        """Open a validated vector matrix and word-to-row index."""
        self.index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
        rows, dimensions = self.index["shape"]
        expected_size = rows * dimensions
        if binary_path.stat().st_size != expected_size * np.dtype(np.float16).itemsize:
            raise ValueError("Vector binary size does not match the index shape")
        self.matrix = np.memmap(
            binary_path,
            dtype=np.float16,
            mode="r",
            shape=(rows, dimensions),
        )
        self.words: dict[str, int] = self.index["words"]

    def vector(self, word: str) -> NDArray[np.float32]:
        """Return one normalized vector by its exact vocabulary key."""
        try:
            row = self.words[word]
        except KeyError as error:
            raise KeyError(f"Word is not in this vector store: {word}") from error
        return l2_normalize_rows(np.asarray(self.matrix[row : row + 1]))[0]


def cosine_similarity(store: VectorStore, word_a: str, word_b: str) -> float:
    """Calculate cosine similarity for two words in a shared hybrid space."""
    left = store.vector(word_a)
    right = store.vector(word_b)
    return float(np.clip(np.dot(left, right), -1.0, 1.0))
