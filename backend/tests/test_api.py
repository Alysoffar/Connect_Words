"""API-level gameplay, error, clue, and latency coverage."""

from __future__ import annotations

from datetime import date
from statistics import median
from time import perf_counter

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.graph import canonical_edge_key
from app.main import create_app
from app.normalize.models import ValidationReason, ValidationResult
from app.runtime import (
    CLUE_MIN_GUESSES,
    GameRuntime,
    InMemoryVectorStore,
)
from app.schemas import PuzzleRecord
from app.scoring import score, score_existing_pairs

TEST_SESSION_ID = "00000000-0000-4000-8000-000000000001"


class FakeNormalizer:
    """Small deterministic validator for API contract tests."""

    def __init__(self, words: set[str]) -> None:
        self.words = words

    def normalize_and_validate(self, word: str, lang: str) -> ValidationResult:
        del lang
        normalized = word.strip().lower()
        if normalized in self.words:
            return ValidationResult(True, normalized, word, normalized, None)
        return ValidationResult(
            False,
            normalized,
            word,
            None,
            ValidationReason.NOT_IN_VOCAB,
        )


def make_puzzle() -> PuzzleRecord:
    return PuzzleRecord.model_validate(
        {
            "id": "en-daily-2026-11-26",
            "lang": "en",
            "anchorA": "start",
            "anchorB": "finish",
            "shortestPath": ["start", "bridge1", "bridge2", "bridge3", "finish"],
            "alternativePaths": [
                ["start", "alt1a", "alt1b", "alt1c", "finish"],
                ["start", "alt2a", "alt2b", "alt2c", "finish"],
            ],
            "difficulty": "easy",
            "date": "2026-11-26",
        }
    )


def make_runtime(
    words: list[str],
    lexical_edges: dict[tuple[str, str], float],
    *,
    threshold: float = 0.5,
    puzzle: PuzzleRecord | None = None,
) -> GameRuntime:
    puzzle = puzzle or make_puzzle()
    vector_store = InMemoryVectorStore(
        np.eye(len(words), dtype=np.float16),
        {word: index for index, word in enumerate(words)},
    )
    return GameRuntime(
        normalizer=FakeNormalizer(set(words)),
        vector_stores={"en": vector_store},
        lexical_scores={
            "en": {
                canonical_edge_key(first, second): weight
                for (first, second), weight in lexical_edges.items()
            }
        },
        thresholds={"en": threshold},
        daily_puzzles={"en": [puzzle]},
        practice_puzzles={"en": [puzzle]},
        today=lambda: date(2026, 11, 26),
    )


def solve_runtime() -> GameRuntime:
    words = [
        "start",
        "finish",
        "bridge1",
        "bridge2",
        "bridge3",
        "alt1a",
        "alt1b",
        "alt1c",
        "alt2a",
        "alt2b",
        "alt2c",
        "near",
    ]
    edges = {
        ("start", "bridge1"): 0.9,
        ("bridge1", "bridge2"): 0.9,
        ("bridge2", "bridge3"): 0.9,
        ("bridge3", "finish"): 0.9,
        ("start", "alt1a"): 0.85,
        ("alt1a", "alt1b"): 0.85,
        ("alt1b", "alt1c"): 0.85,
        ("alt1c", "finish"): 0.85,
        ("start", "alt2a"): 0.8,
        ("alt2a", "alt2b"): 0.8,
        ("alt2b", "alt2c"): 0.8,
        ("alt2c", "finish"): 0.8,
        ("start", "near"): 0.47,
    }
    return make_runtime(words, edges)


def test_daily_and_practice_endpoints_return_expected_contract() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        daily = client.get(
            f"/api/puzzle/daily?lang=en&sessionId={TEST_SESSION_ID}"
        )
        practice = client.get(
            f"/api/puzzle/practice?lang=en&sessionId={TEST_SESSION_ID}"
        )

    assert daily.status_code == 200
    assert daily.json() == {
        "puzzleId": "en-daily-2026-11-26",
        "lang": "en",
        "anchorA": "start",
        "anchorB": "finish",
        "threshold": 0.5,
    }
    assert practice.status_code == 200
    assert practice.json()["puzzleId"] == daily.json()["puzzleId"]


def test_unsupported_language_returns_a_structured_error() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        response = client.get("/api/puzzle/daily?lang=fr")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_full_end_to_end_solve_returns_path_and_score() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        board = ["start", "finish"]
        response = None
        for word in ("bridge1", "bridge2", "bridge3"):
            response = client.post(
                "/api/guess",
                json={
                    "puzzleId": "en-daily-2026-11-26",
                    "sessionId": TEST_SESSION_ID,
                    "mode": "daily",
                    "lang": "en",
                    "word": word,
                    "currentBoardWords": board,
                },
            )
            assert response.status_code == 200
            board.append(word)

    assert response is not None
    solved = response.json()
    assert solved["connected"] is True
    assert solved["path"] == ["start", "bridge1", "bridge2", "bridge3", "finish"]
    assert solved["score"] == 52.67
    assert solved["edges"]


def test_near_miss_is_flagged_and_does_not_connect_board() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        response = client.post(
            "/api/guess",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "word": "near",
                "currentBoardWords": ["start", "finish"],
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["connected"] is False
    assert any(
        edge["nearMiss"] and {edge["wordA"], edge["wordB"]} == {"near", "start"}
        for edge in result["edges"]
    )


def test_duplicate_guess_has_a_structured_conflict_error() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        response = client.post(
            "/api/guess",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "word": "start",
                "currentBoardWords": ["start", "finish"],
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DUPLICATE_WORD"


def test_invalid_guess_is_logged_with_vocabulary_rejection_reason() -> None:
    application = create_app(solve_runtime())
    with TestClient(application) as client:
        response = client.post(
            "/api/guess",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "word": "unknownword",
                "currentBoardWords": ["start", "finish"],
            },
        )
        rejected = application.state.analytics.rejected_words("en")

    assert response.status_code == 422
    assert response.json()["error"]["details"]["reason"] == "not_in_vocab"
    assert rejected[-1]["word"] == "unknownword"
    assert rejected[-1]["reason"] == "not_in_vocab"


def test_guess_rejects_unaccepted_words_in_submitted_board() -> None:
    application = create_app(solve_runtime())
    with TestClient(application) as client:
        response = client.post(
            "/api/guess",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "word": "bridge2",
                "currentBoardWords": ["start", "finish", "bridge1"],
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "BOARD_MISMATCH"


def test_stats_endpoint_counts_started_games() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        client.get(
            f"/api/puzzle/daily?lang=en&sessionId={TEST_SESSION_ID}"
        )
        response = client.get("/api/stats?lang=en")

    assert response.status_code == 200
    payload = response.json()
    assert payload["global"]["startedGames"] == 1
    assert payload["global"]["solves"] == 0
    assert payload["languages"][0]["lang"] == "en"


def test_admin_playtest_requires_token_and_submits_ten_puzzle_ratings() -> None:
    runtime = solve_runtime()
    runtime.practice_puzzles = {
        "en": [
            make_puzzle().model_copy(
                update={
                    "id": f"en-practice-{index:02d}",
                    "date": None,
                }
            )
            for index in range(10)
        ]
    }
    application = create_app(runtime, admin_token="test-admin-token")
    headers = {"Authorization": "Bearer test-admin-token"}
    with TestClient(application) as client:
        unauthorized = client.get("/api/admin/playtest/puzzles?lang=en")
        selected = client.get(
            "/api/admin/playtest/puzzles?lang=en",
            headers=headers,
        )
        puzzles = selected.json()["puzzles"]
        response = client.post(
            "/api/admin/playtest/feedback",
            headers=headers,
            json={
                "lang": "en",
                "isNativeSpeaker": True,
                "responses": [
                    {
                        "puzzleId": puzzle["puzzleId"],
                        "rating": 4,
                        "feedback": "Clear clue path.",
                    }
                    for puzzle in puzzles
                ],
            },
        )

    assert unauthorized.status_code == 401
    assert selected.status_code == 200
    assert len(puzzles) == 10
    assert response.status_code == 200
    assert response.json()["submitted"] == 10


def test_guess_rejects_a_full_fifty_word_board() -> None:
    fillers = [f"filler{index}" for index in range(48)]
    words = ["start", "finish", "new", *fillers]
    with TestClient(create_app(make_runtime(words, {}))) as client:
        response = client.post(
            "/api/guess",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "word": "new",
                "currentBoardWords": ["start", "finish", *fillers],
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "BOARD_FULL"


def test_guess_rejects_board_over_fifty_words() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        response = client.post(
            "/api/guess",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "word": "bridge1",
                "currentBoardWords": ["start", "finish", *[f"extra{i}" for i in range(49)]],
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "BOARD_FULL"


def test_stronger_new_edge_reports_displaced_fifth_edge() -> None:
    words = ["hub", "target", "a", "b", "c", "d", "new"]
    edges = {
        ("hub", "target"): 0.5,
        ("hub", "a"): 0.6,
        ("hub", "b"): 0.7,
        ("hub", "c"): 0.8,
        ("hub", "d"): 0.9,
        ("hub", "new"): 0.55,
    }
    runtime = make_runtime(words, edges, puzzle=PuzzleRecord.model_validate({
        "id": "prune-test",
        "lang": "en",
        "anchorA": "hub",
        "anchorB": "target",
        "shortestPath": ["hub", "target"],
        "alternativePaths": [],
        "difficulty": "easy",
    }))

    with TestClient(create_app(runtime)) as client:
        board = ["hub", "target"]
        for word in ("a", "b", "c", "d"):
            accepted = client.post(
                "/api/guess",
                json={
                    "puzzleId": "prune-test",
                    "sessionId": TEST_SESSION_ID,
                    "mode": "daily",
                    "lang": "en",
                    "word": word,
                    "currentBoardWords": board,
                },
            )
            assert accepted.status_code == 200
            board.append(word)
        response = client.post(
            "/api/guess",
            json={
                "puzzleId": "prune-test",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "word": "new",
                "currentBoardWords": board,
            },
        )

    assert response.status_code == 200
    pruned = response.json()["prunedEdges"]
    assert len(pruned) == 1
    assert {pruned[0]["wordA"], pruned[0]["wordB"]} == {"hub", "target"}
    assert pruned[0]["retained"] is False


def test_clue_is_locked_until_twenty_accepted_guesses_then_uses_valid_path() -> None:
    fillers = [f"filler{index}" for index in range(CLUE_MIN_GUESSES)]
    words = [
        "start",
        "finish",
        "bridge1",
        "bridge2",
        "bridge3",
        "alt1a",
        "alt1b",
        "alt1c",
        "alt2a",
        "alt2b",
        "alt2c",
        *fillers,
    ]
    with TestClient(create_app(make_runtime(words, {}))) as client:
        locked = client.post(
            "/api/clue",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "currentBoardWords": ["start", "finish"],
            },
        )
        board = ["start", "finish"]
        for word in fillers:
            accepted = client.post(
                "/api/guess",
                json={
                    "puzzleId": "en-daily-2026-11-26",
                    "sessionId": TEST_SESSION_ID,
                    "mode": "daily",
                    "lang": "en",
                    "word": word,
                    "currentBoardWords": board,
                },
            )
            assert accepted.status_code == 200
            board.append(word)
        unlocked = client.post(
            "/api/clue",
            json={
                "puzzleId": "en-daily-2026-11-26",
                "sessionId": TEST_SESSION_ID,
                "mode": "daily",
                "lang": "en",
                "currentBoardWords": board,
            },
        )

    assert locked.status_code == 409
    assert locked.json()["error"]["code"] == "CLUE_LOCKED"
    assert unlocked.status_code == 200
    assert unlocked.json()["word"] in {
        "bridge1",
        "bridge3",
        "alt1a",
        "alt1c",
        "alt2a",
        "alt2c",
    }


def test_rate_limit_returns_structured_429() -> None:
    with TestClient(create_app(solve_runtime(), rate_limit=1)) as client:
        first = client.get("/api/puzzle/daily?lang=en")
        second = client.get("/api/puzzle/daily?lang=en")

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "RATE_LIMITED"


def test_similarity_computation_stays_under_five_milliseconds() -> None:
    rng = np.random.default_rng(42)
    words = [f"term{index}" for index in range(50)]
    vectors = rng.normal(size=(50, 1324)).astype(np.float16)
    store = InMemoryVectorStore(
        vectors,
        {word: index for index, word in enumerate(words)},
    )
    lexical: dict[tuple[str, str], float] = {}
    board = words[:-1]
    candidate = words[-1]

    def compute() -> None:
        prior = score_existing_pairs(
            board,
            vector_store=store,
            lexical_scores=lexical,
        )
        candidate_scores = {
            canonical_edge_key(candidate, word): score(
                "en",
                candidate,
                word,
                vector_store=store,
                lexical_scores=lexical,
            )
            for word in board
        }
        assert len(prior) == 49 * 48 // 2
        assert len(candidate_scores) == 49

    for _ in range(3):
        compute()
    timings = []
    for _ in range(15):
        start = perf_counter()
        compute()
        timings.append(perf_counter() - start)

    assert median(timings) < 0.005


def test_batched_prior_scores_match_single_pair_score() -> None:
    runtime = solve_runtime()
    store = runtime.vector_stores["en"]
    words = ["start", "finish", "bridge1", "bridge2"]
    batched = score_existing_pairs(
        words,
        vector_store=store,
        lexical_scores=runtime.lexical_scores["en"],
    )

    for (word_a, word_b), expected in batched.items():
        assert score(
            "en",
            word_a,
            word_b,
            vector_store=store,
            lexical_scores=runtime.lexical_scores["en"],
        ) == pytest.approx(expected)


def test_openapi_spec_includes_game_endpoints() -> None:
    with TestClient(create_app(solve_runtime())) as client:
        schema = client.get("/openapi.json").json()

    assert "/api/puzzle/daily" in schema["paths"]
    assert "/api/puzzle/practice" in schema["paths"]
    assert "/api/guess" in schema["paths"]
    assert "/api/clue" in schema["paths"]
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    assert schema["paths"]["/api/admin/playtest/puzzles"]["get"]["security"]
