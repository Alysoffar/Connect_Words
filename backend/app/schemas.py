"""Pydantic request and response models for the public game API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

LanguageCode = str


class APIModel(BaseModel):
    """Common strict model settings."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class GuessRequest(APIModel):
    """One word submission with the client's current board."""

    puzzle_id: str = Field(alias="puzzleId", min_length=1)
    session_id: UUID = Field(alias="sessionId")
    mode: Literal["daily", "practice"]
    lang: LanguageCode
    word: str = Field(min_length=1, max_length=128)
    current_board_words: list[str] = Field(alias="currentBoardWords")


class ClueRequest(APIModel):
    """Clue request for a puzzle and current board."""

    puzzle_id: str = Field(alias="puzzleId", min_length=1)
    session_id: UUID = Field(alias="sessionId")
    mode: Literal["daily", "practice"]
    lang: LanguageCode
    current_board_words: list[str] = Field(alias="currentBoardWords")


class EdgeResponse(APIModel):
    """Scored edge returned for the submitted word."""

    word_a: str = Field(alias="wordA")
    word_b: str = Field(alias="wordB")
    weight: float
    near_miss: bool = Field(alias="nearMiss")
    retained: bool


class GuessResponse(APIModel):
    """Validation, edge, win, and scoring results."""

    puzzle_id: str = Field(alias="puzzleId")
    lang: LanguageCode
    word: str
    normalized: str
    edges: list[EdgeResponse]
    pruned_edges: list[EdgeResponse] = Field(alias="prunedEdges")
    connected: bool
    path: list[str] | None
    score: float | None


class PuzzleResponse(APIModel):
    """Anchor-only public puzzle details."""

    puzzle_id: str = Field(alias="puzzleId")
    lang: LanguageCode
    anchor_a: str = Field(alias="anchorA")
    anchor_b: str = Field(alias="anchorB")
    threshold: float


class ClueResponse(APIModel):
    """One unused word from a known valid path."""

    puzzle_id: str = Field(alias="puzzleId")
    word: str
    from_word: str = Field(alias="fromWord")


class APIErrorDetail(APIModel):
    """Stable error code, message, and optional request context."""

    code: str
    message: str
    details: dict[str, object] | list[dict[str, object]] | None = None


class APIErrorResponse(APIModel):
    """Envelope used by expected API failures."""

    error: APIErrorDetail


class PuzzleRecord(APIModel):
    """Persisted puzzle data loaded into process memory."""

    id: str
    lang: LanguageCode
    anchor_a: str = Field(alias="anchorA")
    anchor_b: str = Field(alias="anchorB")
    shortest_path: list[str] = Field(alias="shortestPath")
    alternative_paths: list[list[str]] = Field(alias="alternativePaths")
    difficulty: str
    date: str | None = None


class PuzzleCollection(APIModel):
    """Per-language generated daily queue and practice pool."""

    lang: LanguageCode
    daily_queue: list[PuzzleRecord]
    practice_pool: list[PuzzleRecord]


class StatsAggregate(APIModel):
    """Solve metrics across all games or one configured language."""

    lang: LanguageCode | None
    started_games: int = Field(alias="startedGames")
    solves: int
    solve_rate: float = Field(alias="solveRate")
    average_path_length: float | None = Field(alias="averagePathLength")
    average_guesses: float | None = Field(alias="averageGuesses")


class FastestSolve(APIModel):
    """Privacy-safe leaderboard row with no session identifier."""

    place: int
    lang: LanguageCode
    path_length: int = Field(alias="pathLength")
    guess_count: int = Field(alias="guessCount")
    duration_ms: int = Field(alias="durationMs")
    solved_at: datetime = Field(alias="solvedAt")


class StatsResponse(APIModel):
    """Global, per-language, and current-day solve statistics."""

    global_stats: StatsAggregate = Field(alias="global")
    languages: list[StatsAggregate]
    today_fastest_solves: list[FastestSolve] = Field(alias="todayFastestSolves")
    slow_guess_threshold: float = Field(alias="slowGuessThreshold")


class PlaytestPuzzle(APIModel):
    """Minimal puzzle details rendered by the internal feedback form."""

    puzzle_id: str = Field(alias="puzzleId")
    lang: LanguageCode
    anchor_a: str = Field(alias="anchorA")
    anchor_b: str = Field(alias="anchorB")


class PlaytestPuzzlesResponse(APIModel):
    """Ten selected puzzles for one native-speaker language review."""

    lang: LanguageCode
    puzzles: list[PlaytestPuzzle] = Field(min_length=10, max_length=10)


class PlaytestFeedbackItem(APIModel):
    """One rating and optional note for a reviewed puzzle."""

    puzzle_id: str = Field(alias="puzzleId", min_length=1)
    rating: int = Field(ge=1, le=5)
    feedback: str = Field(default="", max_length=1000)


class PlaytestFeedbackRequest(APIModel):
    """Native-speaker feedback for exactly ten puzzles in one language."""

    lang: LanguageCode
    is_native_speaker: bool = Field(alias="isNativeSpeaker")
    responses: list[PlaytestFeedbackItem] = Field(min_length=10, max_length=10)


class PlaytestFeedbackResponse(APIModel):
    """Feedback batch insertion result."""

    lang: LanguageCode
    submitted: int


