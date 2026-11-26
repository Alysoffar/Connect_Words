"""Tests for anonymous gameplay persistence and stats aggregation."""

from datetime import date

from app.analytics import AnalyticsStore
from app.schemas import PuzzleRecord


def _puzzle(puzzle_id: str = "en-daily-2026-11-26") -> PuzzleRecord:
    return PuzzleRecord.model_validate(
        {
            "id": puzzle_id,
            "lang": "en",
            "anchorA": "start",
            "anchorB": "finish",
            "shortestPath": ["start", "bridge", "finish"],
            "alternativePaths": [],
            "difficulty": "easy",
            "date": "2026-11-26",
        }
    )


def test_stats_aggregate_solve_rate_path_length_and_guesses() -> None:
    puzzle = _puzzle()
    with AnalyticsStore(":memory:") as analytics:
        analytics.start_session("session-solved", puzzle, "daily", started_at=100)
        analytics.record_guess(
            session_id="session-solved",
            puzzle=puzzle,
            word="bridge",
            normalized_word="bridge",
            accepted=True,
            created_at=105,
        )
        analytics.record_solve(
            session_id="session-solved",
            puzzle=puzzle,
            path_length=2,
            guess_count=36,
            score=88,
            solved_at=120,
        )
        analytics.start_session("session-open", puzzle, "daily", started_at=110)
        analytics.record_guess(
            session_id="session-open",
            puzzle=puzzle,
            word="unkown",
            normalized_word=None,
            accepted=False,
            rejection_reason="not_in_vocab",
            created_at=115,
        )

        result = analytics.stats(
            ("en", "de"),
            today=date(1970, 1, 1),
        )
        slow = analytics.slow_puzzles_csv("en", threshold=35)
        rejected = analytics.rejected_words_csv("en")

    assert result["global"]["started_games"] == 2
    assert result["global"]["solves"] == 1
    assert result["global"]["solve_rate"] == 0.5
    assert result["global"]["average_path_length"] == 2
    assert result["global"]["average_guesses"] == 36
    assert result["languages"][0]["lang"] == "en"
    assert result["languages"][1]["started_games"] == 0
    assert result["today_fastest_solves"][0]["duration_ms"] == 20_000
    assert "en-daily-2026-11-26" in slow
    assert "unkown,not_in_vocab" in rejected


def test_fastest_solves_only_include_daily_games_and_apply_language_filter() -> None:
    puzzle = _puzzle()
    with AnalyticsStore(":memory:") as analytics:
        analytics.start_session("daily", puzzle, "daily", started_at=100)
        analytics.record_solve(
            session_id="daily",
            puzzle=puzzle,
            path_length=2,
            guess_count=3,
            score=88,
            solved_at=120,
        )
        analytics.start_session("practice", puzzle, "practice", started_at=100)
        analytics.record_solve(
            session_id="practice",
            puzzle=puzzle,
            path_length=2,
            guess_count=3,
            score=88,
            solved_at=101,
        )
        fastest = analytics.stats(
            ("en", "de"),
            today=date(1970, 1, 1),
        )["today_fastest_solves"]
        filtered = analytics.stats(
            ("en", "de"),
            today=date(1970, 1, 1),
            leaderboard_language="de",
        )["today_fastest_solves"]

    assert len(fastest) == 1
    assert fastest[0]["duration_ms"] == 20_000
    assert filtered == []


def test_expected_board_contains_only_server_accepted_guesses() -> None:
    puzzle = _puzzle()
    with AnalyticsStore(":memory:") as analytics:
        analytics.start_session("session", puzzle, "daily", started_at=100)
        analytics.record_guess(
            session_id="session",
            puzzle=puzzle,
            word="bridge",
            normalized_word="bridge",
            accepted=True,
            created_at=105,
        )
        analytics.record_guess(
            session_id="session",
            puzzle=puzzle,
            word="madeup",
            normalized_word=None,
            accepted=False,
            rejection_reason="not_in_vocab",
            created_at=106,
        )

        expected = analytics.expected_board("session", puzzle)

    assert expected == ["start", "finish", "bridge"]
