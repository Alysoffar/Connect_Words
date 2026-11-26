"""Startup-loaded game assets and stateless guess/clue operations."""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .errors import APIError
from .graph import (
    Edge,
    canonical_edge_key,
    enforce_top_five,
    final_score,
    shortest_path,
)
from .normalize import NormalizationService, ValidationResult
from .normalize.dependencies import REPOSITORY_ROOT, load_language_settings
from .schemas import PuzzleCollection, PuzzleRecord
from .scoring import score, score_existing_pairs

LOGGER = logging.getLogger(__name__)
BOARD_LIMIT = 50
NEAR_MISS_MARGIN = 0.05
CLUE_MIN_GUESSES = 20


def utc_today() -> date:
    """Return the current UTC calendar date for daily puzzle selection."""
    return datetime.now(timezone.utc).date()


class WordValidator(Protocol):
    """Interface required from the process-wide normalizer."""

    indexes: Mapping[str, VocabularyTerms]

    def normalize_and_validate(self, word: str, lang: str) -> ValidationResult: ...


class VocabularyTerms(Protocol):
    """Canonical terms exposed by a language vocabulary index."""

    terms: set[str]


class InMemoryVectorStore:
    """Validated float16 vectors and normalized-vector memoization."""

    def __init__(self, matrix: NDArray[np.float16], words: Mapping[str, int]) -> None:
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Vector matrix must be finite and two-dimensional")
        if len(words) != matrix.shape[0]:
            raise ValueError("Vector index size does not match the matrix")
        self.matrix = matrix
        self.words = dict(words)
        self._cached_vector = lru_cache(maxsize=4096)(self._load_vector)

    @classmethod
    def from_files(cls, binary_path: Path, index_path: Path) -> "InMemoryVectorStore":
        if not binary_path.is_file():
            raise FileNotFoundError(f"Missing vector matrix: {binary_path}")
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing vector index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shape = index.get("shape")
        words = index.get("words")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(size, int) and size > 0 for size in shape)
            or not isinstance(words, dict)
        ):
            raise ValueError(f"Invalid vector index: {index_path}")
        rows, dimensions = shape
        expected_bytes = rows * dimensions * np.dtype(np.float16).itemsize
        if binary_path.stat().st_size != expected_bytes:
            raise ValueError(f"Vector binary size does not match index: {binary_path}")
        matrix = np.fromfile(binary_path, dtype=np.float16).reshape(shape)
        if (
            len(words) != rows
            or any(not isinstance(word, str) for word in words)
            or set(words.values()) != set(range(rows))
        ):
            raise ValueError(f"Invalid vocabulary row map: {index_path}")
        return cls(matrix, words)

    def _load_vector(self, word: str) -> NDArray[np.float32]:
        try:
            row = self.words[word]
        except KeyError as error:
            raise KeyError(f"Word is not in this vector store: {word}") from error
        vector = np.asarray(self.matrix[row], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Invalid vector for word: {word}")
        vector /= norm
        vector.flags.writeable = False
        return vector

    def vector(self, word: str) -> NDArray[np.float32]:
        """Return a cached normalized float32 vector for one term."""
        return self._cached_vector(word)


@dataclass(slots=True)
class GuessResult:
    """Pure game-engine result before Pydantic response serialization."""

    word: str
    normalized: str
    edges: list[Edge]
    retained_edge_keys: set[tuple[str, str]]
    pruned_edges: list[Edge]
    connected: bool
    connected_path: list[str] | None
    final_score: float | None


@dataclass(slots=True)
class GameRuntime:
    """Per-process immutable assets; player state is supplied on every request."""

    normalizer: WordValidator
    vector_stores: Mapping[str, InMemoryVectorStore]
    lexical_scores: Mapping[str, Mapping[tuple[str, str], float]]
    thresholds: Mapping[str, float]
    daily_puzzles: Mapping[str, Sequence[PuzzleRecord]]
    practice_puzzles: Mapping[str, Sequence[PuzzleRecord]]
    today: Callable[[], date] = utc_today

    @classmethod
    def from_repository(
        cls,
        *,
        data_root: Path | None = None,
        repository_root: Path = REPOSITORY_ROOT,
    ) -> "GameRuntime":
        """Load and validate vectors, WordNet scores, puzzles, and vocab indexes."""
        data_root = data_root or repository_root / "data"
        normalizer = NormalizationService.from_repository(
            data_root=data_root,
            blocklist_root=repository_root / "shared" / "blocklists",
        )
        language_settings = load_language_settings()
        thresholds: dict[str, float] = {}
        for language, settings in language_settings.items():
            threshold = settings["threshold"]
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise ValueError(
                    f"Connection threshold for {language} must be calibrated "
                    "to a value in [0, 1]"
                )
            thresholds[language] = float(threshold)

        vector_stores = {
            language: InMemoryVectorStore.from_files(
                data_root / "vectors" / f"{language}.f16.bin",
                data_root / "vectors" / f"{language}.index.json",
            )
            for language in language_settings
        }
        for language, store in vector_stores.items():
            missing_vectors = normalizer.indexes[language].terms - store.words.keys()
            if missing_vectors:
                raise ValueError(
                    f"Vector index for {language} is missing "
                    f"{len(missing_vectors)} vocabulary terms"
                )
        lexical_scores = cls._load_lexical_scores(data_root / "linkwords.sqlite")

        daily_puzzles: dict[str, list[PuzzleRecord]] = {}
        practice_puzzles: dict[str, list[PuzzleRecord]] = {}
        seen_ids: set[str] = set()
        for language in language_settings:
            path = data_root / "puzzles" / f"{language}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing puzzle collection: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            collection = PuzzleCollection.model_validate(payload)
            if collection.lang != language:
                raise ValueError(f"Puzzle collection language mismatch: {path}")
            if not collection.daily_queue or not collection.practice_pool:
                raise ValueError(
                    f"Puzzle collection must include daily and practice puzzles: {path}"
                )
            daily_puzzles[language] = collection.daily_queue
            practice_puzzles[language] = collection.practice_pool
            for puzzle in (*collection.daily_queue, *collection.practice_pool):
                if puzzle.id in seen_ids:
                    raise ValueError(f"Duplicate puzzle ID: {puzzle.id}")
                if puzzle.lang != language:
                    raise ValueError(f"Puzzle language mismatch for {puzzle.id}")
                if puzzle.id.strip() == "":
                    raise ValueError("Puzzle IDs must not be empty")
                if (
                    not puzzle.shortest_path
                    or puzzle.shortest_path[0] != puzzle.anchor_a
                    or puzzle.shortest_path[-1] != puzzle.anchor_b
                ):
                    raise ValueError(f"Invalid shortest path for {puzzle.id}")
                paths = (puzzle.shortest_path, *puzzle.alternative_paths)
                vocabulary = normalizer.indexes[language].terms
                if any(
                    len(route) < 2
                    or route[0] != puzzle.anchor_a
                    or route[-1] != puzzle.anchor_b
                    or any(word not in vocabulary for word in route)
                    for route in paths
                ):
                    raise ValueError(f"Invalid or out-of-vocabulary path for {puzzle.id}")
                seen_ids.add(puzzle.id)

        LOGGER.info(
            "Loaded game assets: %d vector stores, %d lexical pairs, %d puzzles",
            len(vector_stores),
            sum(map(len, lexical_scores.values())),
            sum(map(len, daily_puzzles.values()))
            + sum(map(len, practice_puzzles.values())),
        )
        return cls(
            normalizer=normalizer,
            vector_stores=vector_stores,
            lexical_scores=lexical_scores,
            thresholds=thresholds,
            daily_puzzles=daily_puzzles,
            practice_puzzles=practice_puzzles,
        )

    @staticmethod
    def _load_lexical_scores(
        database_path: Path,
    ) -> dict[str, dict[tuple[str, str], float]]:
        if not database_path.is_file():
            raise FileNotFoundError(f"Missing WordNet database: {database_path}")
        result: dict[str, dict[tuple[str, str], float]] = {}
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                """
                SELECT lang, word_a, word_b, MAX(boost_score)
                FROM synonyms
                GROUP BY lang, word_a, word_b
                """
            )
            for language, word_a, word_b, weight in rows:
                result.setdefault(language, {})[(word_a, word_b)] = float(weight)
        return result

    def get_puzzle(self, puzzle_id: str, language: str) -> PuzzleRecord:
        """Return a puzzle only when both its ID and language match."""
        self.require_language(language)
        pools = (
            *self.daily_puzzles.get(language, ()),
            *self.practice_puzzles.get(language, ()),
        )
        for puzzle in pools:
            if puzzle.id == puzzle_id:
                return puzzle
        raise APIError(404, "PUZZLE_NOT_FOUND", "Puzzle was not found for this language")

    def require_language(self, language: str) -> None:
        """Reject languages not present in the shared runtime configuration."""
        if language not in self.thresholds:
            raise APIError(
                422,
                "UNSUPPORTED_LANGUAGE",
                "Language is not configured for this service",
            )

    def daily_puzzle(self, language: str) -> PuzzleRecord:
        """Return today's dated puzzle for one language."""
        self.require_language(language)
        target_date = self.today().isoformat()
        for puzzle in self.daily_puzzles.get(language, ()):
            if puzzle.date == target_date:
                return puzzle
        raise APIError(404, "DAILY_PUZZLE_NOT_FOUND", "No daily puzzle is available today")

    def practice_puzzle(self, language: str) -> PuzzleRecord:
        """Select one practice puzzle without retaining player state."""
        self.require_language(language)
        puzzles = self.practice_puzzles.get(language, ())
        if not puzzles:
            raise APIError(404, "PRACTICE_POOL_EMPTY", "No practice puzzles are available")
        return random.choice(puzzles)

    def validate_board(
        self,
        puzzle: PuzzleRecord,
        board_words: Sequence[str],
    ) -> list[str]:
        """Normalize and validate a supplied board, including its puzzle anchors."""
        if len(board_words) > BOARD_LIMIT:
            raise APIError(409, "BOARD_FULL", f"Board limit is {BOARD_LIMIT} words")
        normalized_words: list[str] = []
        seen: set[str] = set()
        for word in board_words:
            result = self.normalizer.normalize_and_validate(word, puzzle.lang)
            if not result.ok:
                raise APIError(
                    422,
                    "INVALID_BOARD_WORD",
                    "Current board contains a word that is not valid",
                    details={"reason": result.reason.value if result.reason else None},
                )
            if result.normalized in seen:
                raise APIError(
                    409,
                    "DUPLICATE_BOARD_WORD",
                    "Current board contains duplicate normalized words",
                )
            if result.normalized not in self.vector_stores[puzzle.lang].words:
                raise APIError(
                    422,
                    "WORD_NOT_SCORABLE",
                    "Current board contains a word with no vector",
                )
            seen.add(result.normalized)
            normalized_words.append(result.normalized)

        anchor_a = self.normalizer.normalize_and_validate(puzzle.anchor_a, puzzle.lang)
        anchor_b = self.normalizer.normalize_and_validate(puzzle.anchor_b, puzzle.lang)
        if (
            not anchor_a.ok
            or not anchor_b.ok
            or anchor_a.normalized not in seen
            or anchor_b.normalized not in seen
        ):
            raise APIError(
                422,
                "PUZZLE_ANCHORS_MISSING",
                "Current board must contain both puzzle anchors",
            )
        return normalized_words

    def submit_guess(
        self,
        puzzle: PuzzleRecord,
        word: str,
        current_board_words: Sequence[str],
        *,
        expected_board_words: Sequence[str] | None = None,
    ) -> GuessResult:
        """Apply validation, scoring, edge caps, and win detection to one guess."""
        board = self.validate_board(puzzle, current_board_words)
        if len(board) >= BOARD_LIMIT:
            raise APIError(409, "BOARD_FULL", f"Board limit is {BOARD_LIMIT} words")
        if expected_board_words is not None and board != list(expected_board_words):
            raise APIError(
                409,
                "BOARD_MISMATCH",
                "Board does not match the accepted guesses in this session",
            )
        validation = self.normalizer.normalize_and_validate(word, puzzle.lang)
        if not validation.ok:
            raise APIError(
                422,
                "INVALID_WORD",
                "Word did not pass normalization and vocabulary validation",
                details={"reason": validation.reason.value if validation.reason else None},
            )
        candidate = validation.normalized
        if candidate in board:
            raise APIError(409, "DUPLICATE_WORD", "Word is already on the board")
        store = self.vector_stores[puzzle.lang]
        if candidate not in store.words:
            raise APIError(
                422,
                "WORD_NOT_SCORABLE",
                "Word has no vector in the active language index",
            )
        lexical = self.lexical_scores.get(puzzle.lang, {})
        threshold = self.thresholds[puzzle.lang]
        existing_weights = score_existing_pairs(
            board,
            vector_store=store,
            lexical_scores=lexical,
        )
        before = enforce_top_five(board, existing_weights, threshold)
        candidate_weights = {
            canonical_edge_key(candidate, existing_word): score(
                puzzle.lang,
                candidate,
                existing_word,
                vector_store=store,
                lexical_scores=lexical,
            )
            for existing_word in board
        }
        all_weights = {**existing_weights, **candidate_weights}
        updated_board = [*board, candidate]
        after = enforce_top_five(updated_board, all_weights, threshold)
        selected_keys = {edge.key for edge in after.edges}
        before_keys = {edge.key for edge in before.edges}

        pruned_by_new_word = [
            edge for edge in before.edges if edge.key not in selected_keys
        ]
        pruned_by_cap = [
            edge
            for edge in after.pruned_edges
            if candidate in (edge.word_a, edge.word_b)
        ]
        pruned_edges = pruned_by_new_word + pruned_by_cap
        output_edges: list[Edge] = []
        for key, weight in candidate_weights.items():
            if weight >= threshold or threshold - NEAR_MISS_MARGIN <= weight:
                output_edges.append(Edge(key[0], key[1], weight))

        path = shortest_path(after.adjacency, puzzle.anchor_a, puzzle.anchor_b)
        score_value: float | None = None
        if path is not None:
            path_weights = [
                all_weights[canonical_edge_key(first, second)]
                for first, second in zip(path, path[1:])
            ]
            accepted_guesses = max(1, len(updated_board) - 2)
            score_value = final_score(
                len(path) - 1,
                sum(path_weights) / len(path_weights),
                accepted_guesses,
            )

        LOGGER.debug(
            "Guess graph update: %d prior edges, %d selected, %d pruned",
            len(before_keys),
            len(after.edges),
            len(pruned_edges),
        )
        return GuessResult(
            word=validation.display,
            normalized=candidate,
            edges=output_edges,
            retained_edge_keys=selected_keys,
            pruned_edges=pruned_edges,
            connected=path is not None,
            connected_path=path,
            final_score=score_value,
        )

    def clue(
        self,
        puzzle: PuzzleRecord,
        current_board_words: Sequence[str],
        *,
        expected_board_words: Sequence[str] | None = None,
    ) -> tuple[str, str]:
        """Return an unused neighbor from a stored valid puzzle path."""
        board = self.validate_board(puzzle, current_board_words)
        if expected_board_words is not None and board != list(expected_board_words):
            raise APIError(
                409,
                "BOARD_MISMATCH",
                "Board does not match the accepted guesses in this session",
            )
        accepted_guesses = max(0, len(board) - 2)
        if accepted_guesses < CLUE_MIN_GUESSES:
            raise APIError(
                409,
                "CLUE_LOCKED",
                f"Clue unlocks after {CLUE_MIN_GUESSES} accepted guesses",
                details={"guessesRemaining": CLUE_MIN_GUESSES - accepted_guesses},
            )

        board_set = set(board)
        candidates: set[tuple[str, str]] = set()
        for path in (puzzle.shortest_path, *puzzle.alternative_paths):
            for first, second in zip(path, path[1:]):
                if first in board_set and second not in board_set:
                    candidates.add((second, first))
                if second in board_set and first not in board_set:
                    candidates.add((first, second))
        if not candidates:
            raise APIError(409, "NO_CLUE_AVAILABLE", "No unused path neighbor is available")
        word, from_word = random.choice(sorted(candidates))
        return word, from_word


def make_repository_runtime() -> GameRuntime:
    """Load runtime assets using the configured repository data directory."""
    data_root = Path(os.environ.get("LINKWORDS_DATA_ROOT", REPOSITORY_ROOT / "data"))
    return GameRuntime.from_repository(data_root=data_root)
