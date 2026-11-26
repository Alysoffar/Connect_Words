"""FastAPI application for the LinkWords game service."""

from __future__ import annotations

import logging
import os
import time
import hmac
from collections.abc import AsyncGenerator
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import UUID

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .analytics import AnalyticsStore
from .errors import APIError
from .normalize.dependencies import REPOSITORY_ROOT
from .runtime import GameRuntime, make_repository_runtime
from .schemas import (
    APIErrorResponse,
    ClueRequest,
    ClueResponse,
    EdgeResponse,
    GuessRequest,
    GuessResponse,
    LanguageCode,
    PlaytestFeedbackRequest,
    PlaytestFeedbackResponse,
    PlaytestPuzzlesResponse,
    PlaytestPuzzle,
    PuzzleResponse,
    StatsResponse,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_RATE_LIMIT = 60
DEFAULT_RATE_WINDOW_SECONDS = 60.0


class SlidingWindowLimiter:
    """Thread-safe per-client request limiter for one API process."""

    def __init__(self, maximum: int, window_seconds: float) -> None:
        if maximum < 1 or window_seconds <= 0:
            raise ValueError("Rate-limit count and window must be positive")
        self.maximum = maximum
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_key: str, now: float | None = None) -> bool:
        """Return whether a request is within the client's rolling window."""
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[client_key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.maximum:
                return False
            events.append(current)
            return True


def _error_payload(
    code: str,
    message: str,
    details: object | None = None,
) -> dict[str, object]:
    error: dict[str, object] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


def _guess_rejection_reason(error: APIError) -> str:
    if isinstance(error.details, dict):
        reason = error.details.get("reason")
        if isinstance(reason, str):
            return reason
    return error.code.lower()


def create_app(
    runtime: GameRuntime | None = None,
    *,
    analytics_store: AnalyticsStore | None = None,
    admin_token: str | None = None,
    cors_origins: list[str] | None = None,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    rate_window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
) -> FastAPI:
    """Construct an app; production assets load during lifespan startup."""
    initial_runtime = runtime
    limiter = SlidingWindowLimiter(rate_limit, rate_window_seconds)
    if analytics_store is None:
        configured_path = os.environ.get("LINKWORDS_ANALYTICS_DB")
        if configured_path:
            analytics_path = configured_path
        elif initial_runtime is None:
            analytics_path = str(
                Path(
                    os.environ.get(
                        "LINKWORDS_DATA_ROOT",
                        REPOSITORY_ROOT / "data",
                    )
                )
                / "analytics.sqlite"
            )
        else:
            analytics_path = ":memory:"
        analytics_store = AnalyticsStore(analytics_path)
    configured_admin_token = (
        admin_token
        if admin_token is not None
        else os.environ.get("LINKWORDS_ADMIN_TOKEN", "")
    )
    bearer_scheme = HTTPBearer(auto_error=False)
    try:
        slow_guess_threshold = float(
            os.environ.get("LINKWORDS_SLOW_GUESS_THRESHOLD", "35")
        )
    except ValueError as error:
        raise ValueError("LINKWORDS_SLOW_GUESS_THRESHOLD must be numeric") from error
    if slow_guess_threshold < 1:
        raise ValueError("LINKWORDS_SLOW_GUESS_THRESHOLD must be positive")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        if application.state.runtime is None:
            application.state.runtime = make_repository_runtime()
        try:
            yield
        finally:
            application.state.analytics.close()

    application = FastAPI(
        title="LinkWords API",
        version="1.0.0",
        description=(
            "Multilingual word-association gameplay API. Vocabulary, vectors, "
            "lexical scores, and puzzle records load at startup; anonymous "
            "session analytics and accepted guesses persist in SQLite."
        ),
        lifespan=lifespan,
        openapi_url="/openapi.json",
    )
    application.state.runtime = initial_runtime
    application.state.analytics = analytics_store
    allowed_origins = cors_origins
    if allowed_origins is None:
        allowed_origins = [
            origin.strip()
            for origin in os.environ.get(
                "LINKWORDS_CORS_ORIGINS",
                "http://localhost:3000",
            ).split(",")
            if origin.strip()
        ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @application.middleware("http")
    async def enforce_rate_limit(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            client_key = request.client.host if request.client else "unknown"
            if not limiter.allow(client_key):
                return JSONResponse(
                    status_code=429,
                    content=_error_payload(
                        "RATE_LIMITED",
                        "Too many requests; retry after the rate window",
                    ),
                    headers={"Retry-After": str(int(rate_window_seconds))},
                )
        return await call_next(request)

    @application.exception_handler(APIError)
    async def handle_api_error(_request: Request, error: APIError) -> JSONResponse:
        del _request
        return JSONResponse(
            status_code=error.status_code,
            content=_error_payload(error.code, error.message, error.details),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del _request
        details = [
            {
                "location": [str(part) for part in item["loc"]],
                "message": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "INVALID_REQUEST",
                "Request validation failed",
                details,
            ),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(
        _request: Request,
        error: Exception,
    ) -> JSONResponse:
        del _request
        LOGGER.exception("Unhandled LinkWords API exception", exc_info=error)
        return JSONResponse(
            status_code=500,
            content=_error_payload("INTERNAL_ERROR", "An internal error occurred"),
        )

    def game_runtime(request: Request) -> GameRuntime:
        current = request.app.state.runtime
        if current is None:
            raise APIError(503, "SERVICE_NOT_READY", "Game assets are not loaded")
        return current

    def analytics(request: Request) -> AnalyticsStore:
        return request.app.state.analytics

    def require_admin(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        if not configured_admin_token:
            raise APIError(
                503,
                "ADMIN_DISABLED",
                "Playtest administration is not configured",
            )
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not hmac.compare_digest(
                credentials.credentials,
                configured_admin_token,
            )
        ):
            raise APIError(401, "ADMIN_UNAUTHORIZED", "Admin token is invalid")

    def playtest_puzzles(
        current: GameRuntime,
        language: str,
    ) -> list[PlaytestPuzzle]:
        current.require_language(language)
        unique = {
            puzzle.id: puzzle
            for puzzle in (
                *current.daily_puzzles.get(language, ()),
                *current.practice_puzzles.get(language, ()),
            )
        }
        selected = sorted(unique.values(), key=lambda puzzle: puzzle.id)[:10]
        if len(selected) != 10:
            raise APIError(
                503,
                "PLAYTEST_PUZZLES_UNAVAILABLE",
                f"Ten unique puzzles are required for {language} playtesting",
            )
        return [
            PlaytestPuzzle(
                puzzleId=puzzle.id,
                lang=puzzle.lang,
                anchorA=puzzle.anchor_a,
                anchorB=puzzle.anchor_b,
            )
            for puzzle in selected
        ]

    def puzzle_response(
        current: GameRuntime,
        puzzle_id: str,
        language: LanguageCode,
    ) -> PuzzleResponse:
        puzzle = current.get_puzzle(puzzle_id, language)
        return PuzzleResponse(
            puzzle_id=puzzle.id,
            lang=puzzle.lang,
            anchor_a=puzzle.anchor_a,
            anchor_b=puzzle.anchor_b,
            threshold=current.thresholds[language],
        )

    @application.get(
        "/api/puzzle/daily",
        response_model=PuzzleResponse,
        responses={
            404: {"model": APIErrorResponse},
            422: {"model": APIErrorResponse},
            429: {"model": APIErrorResponse},
        },
    )
    async def daily_puzzle(
        request: Request,
        lang: LanguageCode,
        session_id: UUID | None = Query(default=None, alias="sessionId"),
    ) -> PuzzleResponse:
        current = game_runtime(request)
        active_id = analytics(request).active_daily_puzzle(lang, current.today())
        puzzle = (
            current.get_puzzle(active_id, lang)
            if active_id is not None
            else current.daily_puzzle(lang)
        )
        if session_id is not None:
            analytics(request).start_session(str(session_id), puzzle, "daily")
        return puzzle_response(current, puzzle.id, lang)

    @application.get(
        "/api/puzzle/practice",
        response_model=PuzzleResponse,
        responses={
            404: {"model": APIErrorResponse},
            422: {"model": APIErrorResponse},
            429: {"model": APIErrorResponse},
        },
    )
    async def practice_puzzle(
        request: Request,
        lang: LanguageCode,
        session_id: UUID | None = Query(default=None, alias="sessionId"),
    ) -> PuzzleResponse:
        current = game_runtime(request)
        puzzle = current.practice_puzzle(lang)
        if session_id is not None:
            analytics(request).start_session(str(session_id), puzzle, "practice")
        return puzzle_response(current, puzzle.id, lang)

    @application.post(
        "/api/guess",
        response_model=GuessResponse,
        responses={
            404: {"model": APIErrorResponse},
            409: {"model": APIErrorResponse},
            422: {"model": APIErrorResponse},
            429: {"model": APIErrorResponse},
        },
    )
    async def submit_guess(
        payload: GuessRequest,
        request: Request,
    ) -> GuessResponse:
        current = game_runtime(request)
        puzzle = current.get_puzzle(payload.puzzle_id, payload.lang)
        store = analytics(request)
        session_id = str(payload.session_id)
        store.start_session(session_id, puzzle, payload.mode)
        expected_board: list[str] | None = None
        try:
            expected_board = store.expected_board(session_id, puzzle)
            result = current.submit_guess(
                puzzle,
                payload.word,
                payload.current_board_words,
                expected_board_words=expected_board,
            )
        except APIError as error:
            store.record_guess(
                session_id=session_id,
                puzzle=puzzle,
                word=payload.word,
                normalized_word=None,
                accepted=False,
                rejection_reason=_guess_rejection_reason(error),
                expected_board_words=expected_board,
            )
            raise
        guess_count = store.record_guess(
            session_id=session_id,
            puzzle=puzzle,
            word=payload.word,
            normalized_word=result.normalized,
            accepted=True,
            expected_board_words=expected_board,
        )
        if result.connected and result.connected_path and result.final_score is not None:
            store.record_solve(
                session_id=session_id,
                puzzle=puzzle,
                path_length=len(result.connected_path) - 1,
                guess_count=guess_count,
                score=result.final_score,
            )
        threshold = current.thresholds[payload.lang]
        edges = [
            EdgeResponse(
                word_a=edge.word_a,
                word_b=edge.word_b,
                weight=edge.weight,
                near_miss=edge.weight < threshold,
                retained=edge.weight >= threshold
                and edge.key in result.retained_edge_keys,
            )
            for edge in result.edges
        ]
        pruned = [
            EdgeResponse(
                word_a=edge.word_a,
                word_b=edge.word_b,
                weight=edge.weight,
                near_miss=False,
                retained=False,
            )
            for edge in result.pruned_edges
        ]
        return GuessResponse(
            puzzle_id=puzzle.id,
            lang=payload.lang,
            word=result.word,
            normalized=result.normalized,
            edges=edges,
            pruned_edges=pruned,
            connected=result.connected,
            path=result.connected_path,
            score=result.final_score,
        )

    @application.post(
        "/api/clue",
        response_model=ClueResponse,
        responses={
            404: {"model": APIErrorResponse},
            409: {"model": APIErrorResponse},
            422: {"model": APIErrorResponse},
            429: {"model": APIErrorResponse},
        },
    )
    async def request_clue(payload: ClueRequest, request: Request) -> ClueResponse:
        current = game_runtime(request)
        puzzle = current.get_puzzle(payload.puzzle_id, payload.lang)
        store = analytics(request)
        session_id = str(payload.session_id)
        store.start_session(session_id, puzzle, payload.mode)
        expected_board = store.expected_board(session_id, puzzle)
        word, from_word = current.clue(
            puzzle,
            payload.current_board_words,
            expected_board_words=expected_board,
        )
        return ClueResponse(
            puzzle_id=puzzle.id,
            word=word,
            from_word=from_word,
        )

    @application.get(
        "/api/stats",
        response_model=StatsResponse,
        responses={422: {"model": APIErrorResponse}, 429: {"model": APIErrorResponse}},
    )
    async def game_stats(
        request: Request,
        lang: LanguageCode | None = None,
    ) -> StatsResponse:
        current = game_runtime(request)
        if lang is not None:
            current.require_language(lang)
        payload = analytics(request).stats(
            tuple(current.thresholds),
            today=current.today(),
            leaderboard_language=lang,
            slow_guess_threshold=slow_guess_threshold,
        )
        for item in payload["today_fastest_solves"]:
            item["solved_at"] = datetime.fromtimestamp(
                item["solved_at"],
                tz=timezone.utc,
            )
        return StatsResponse.model_validate(payload)

    @application.get(
        "/api/admin/playtest/puzzles",
        response_model=PlaytestPuzzlesResponse,
        dependencies=[Depends(require_admin)],
        responses={
            401: {"model": APIErrorResponse},
            422: {"model": APIErrorResponse},
            503: {"model": APIErrorResponse},
        },
    )
    async def admin_playtest_puzzles(
        request: Request,
        lang: LanguageCode,
    ) -> PlaytestPuzzlesResponse:
        return PlaytestPuzzlesResponse(lang=lang, puzzles=playtest_puzzles(game_runtime(request), lang))

    @application.post(
        "/api/admin/playtest/feedback",
        response_model=PlaytestFeedbackResponse,
        dependencies=[Depends(require_admin)],
        responses={
            401: {"model": APIErrorResponse},
            422: {"model": APIErrorResponse},
            503: {"model": APIErrorResponse},
        },
    )
    async def submit_playtest_feedback(
        payload: PlaytestFeedbackRequest,
        request: Request,
    ) -> PlaytestFeedbackResponse:
        if not payload.is_native_speaker:
            raise APIError(
                422,
                "NATIVE_SPEAKER_REQUIRED",
                "This playtest feedback form is for native speakers",
            )
        current = game_runtime(request)
        allowed_puzzles = {
            puzzle.puzzle_id
            for puzzle in playtest_puzzles(current, payload.lang)
        }
        submitted_ids = {item.puzzle_id for item in payload.responses}
        if (
            len(submitted_ids) != 10
            or submitted_ids != allowed_puzzles
        ):
            raise APIError(
                422,
                "INVALID_PLAYTEST_BATCH",
                "Feedback must contain one response for each of the ten assigned puzzles",
            )
        analytics(request).submit_feedback(
            payload.lang,
            [
                {
                    "puzzle_id": item.puzzle_id,
                    "rating": item.rating,
                    "feedback": item.feedback,
                }
                for item in payload.responses
            ],
        )
        return PlaytestFeedbackResponse(lang=payload.lang, submitted=10)

    @application.get(
        "/api/admin/playtest/rejected-words.csv",
        dependencies=[Depends(require_admin)],
        responses={401: {"model": APIErrorResponse}, 503: {"model": APIErrorResponse}},
    )
    async def export_rejected_words(
        request: Request,
        lang: LanguageCode,
    ) -> Response:
        current = game_runtime(request)
        current.require_language(lang)
        return Response(
            content=analytics(request).rejected_words_csv(lang),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="linkwords-rejected-{lang}.csv"'
            },
        )

    @application.get(
        "/api/admin/playtest/slow-puzzles.csv",
        dependencies=[Depends(require_admin)],
        responses={401: {"model": APIErrorResponse}, 503: {"model": APIErrorResponse}},
    )
    async def export_slow_puzzles(
        request: Request,
        lang: LanguageCode,
    ) -> Response:
        current = game_runtime(request)
        current.require_language(lang)
        return Response(
            content=analytics(request).slow_puzzles_csv(
                lang,
                slow_guess_threshold,
            ),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="linkwords-slow-puzzles-{lang}.csv"'
            },
        )

    return application


app = create_app()
