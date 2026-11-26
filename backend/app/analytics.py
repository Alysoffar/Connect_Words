"""SQLite persistence for anonymous gameplay and playtest analytics."""

from __future__ import annotations

import csv
import io
import sqlite3
import threading
import time
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Any

from .errors import APIError
from .schemas import PuzzleRecord


class AnalyticsStore:
    """Persist anonymous sessions, guess outcomes, solves, and feedback."""

    def __init__(self, database_path: str | Path) -> None:
        path = str(database_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            path,
            check_same_thread=False,
            timeout=30,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT NOT NULL,
                puzzle_id TEXT NOT NULL,
                lang TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('daily', 'practice')),
                started_at REAL NOT NULL,
                PRIMARY KEY (session_id, puzzle_id)
            );
            CREATE TABLE IF NOT EXISTS guesses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                puzzle_id TEXT NOT NULL,
                lang TEXT NOT NULL,
                word TEXT NOT NULL,
                normalized_word TEXT,
                accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
                rejection_reason TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id, puzzle_id)
                    REFERENCES sessions(session_id, puzzle_id)
            );
            CREATE INDEX IF NOT EXISTS guesses_session_order
                ON guesses(session_id, puzzle_id, accepted, id);
            CREATE TABLE IF NOT EXISTS solves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                puzzle_id TEXT NOT NULL,
                lang TEXT NOT NULL,
                path_length INTEGER NOT NULL,
                guess_count INTEGER NOT NULL,
                score REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
                solved_at REAL NOT NULL,
                UNIQUE(session_id, puzzle_id),
                FOREIGN KEY (session_id, puzzle_id)
                    REFERENCES sessions(session_id, puzzle_id)
            );
            CREATE INDEX IF NOT EXISTS solves_language_time
                ON solves(lang, solved_at, duration_ms);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lang TEXT NOT NULL,
                puzzle_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                feedback TEXT NOT NULL,
                native_speaker INTEGER NOT NULL CHECK (native_speaker = 1),
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS active_daily_puzzles (
                lang TEXT NOT NULL,
                active_date TEXT NOT NULL,
                puzzle_id TEXT NOT NULL,
                promoted_at REAL NOT NULL,
                PRIMARY KEY (lang, active_date)
            );
            """
        )

    def close(self) -> None:
        """Close the store connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "AnalyticsStore":
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def start_session(
        self,
        session_id: str,
        puzzle: PuzzleRecord,
        mode: str,
        *,
        started_at: float | None = None,
    ) -> None:
        """Create a puzzle session once, preserving its original start time."""
        if mode not in {"daily", "practice"}:
            raise ValueError(f"Unsupported session mode: {mode}")
        with self._lock:
            existing = self._connection.execute(
                """
                SELECT lang FROM sessions
                WHERE session_id = ? AND puzzle_id = ?
                """,
                (session_id, puzzle.id),
            ).fetchone()
            if existing is not None:
                if existing["lang"] != puzzle.lang:
                    raise APIError(
                        409,
                        "SESSION_LANGUAGE_MISMATCH",
                        "This anonymous session belongs to another language for the puzzle",
                    )
                return
            self._connection.execute(
                """
                INSERT INTO sessions(session_id, puzzle_id, lang, mode, started_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    puzzle.id,
                    puzzle.lang,
                    mode,
                    started_at if started_at is not None else time.time(),
                ),
            )

    def expected_board(
        self,
        session_id: str,
        puzzle: PuzzleRecord,
    ) -> list[str]:
        """Return anchors plus accepted guesses, in their original order."""
        with self._lock:
            session = self._connection.execute(
                """
                SELECT lang FROM sessions
                WHERE session_id = ? AND puzzle_id = ?
                """,
                (session_id, puzzle.id),
            ).fetchone()
            if session is None:
                raise APIError(
                    409,
                    "SESSION_NOT_STARTED",
                    "Load this puzzle before submitting guesses",
                )
            if session["lang"] != puzzle.lang:
                raise APIError(
                    409,
                    "SESSION_LANGUAGE_MISMATCH",
                    "This anonymous session belongs to another language for the puzzle",
                )
            accepted = self._connection.execute(
                """
                SELECT normalized_word FROM guesses
                WHERE session_id = ? AND puzzle_id = ? AND accepted = 1
                ORDER BY id
                """,
                (session_id, puzzle.id),
            ).fetchall()
        return [
            puzzle.anchor_a,
            puzzle.anchor_b,
            *(row["normalized_word"] for row in accepted),
        ]

    def record_guess(
        self,
        *,
        session_id: str,
        puzzle: PuzzleRecord,
        word: str,
        normalized_word: str | None,
        accepted: bool,
        rejection_reason: str | None = None,
        expected_board_words: list[str] | None = None,
        created_at: float | None = None,
    ) -> int:
        """Record one accepted or rejected submission and return accepted count."""
        timestamp = created_at if created_at is not None else time.time()
        board_conflict = False
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if expected_board_words is not None:
                    accepted_words = self._connection.execute(
                        """
                        SELECT normalized_word FROM guesses
                        WHERE session_id = ? AND puzzle_id = ? AND accepted = 1
                        ORDER BY id
                        """,
                        (session_id, puzzle.id),
                    ).fetchall()
                    current_board = [
                        puzzle.anchor_a,
                        puzzle.anchor_b,
                        *(row["normalized_word"] for row in accepted_words),
                    ]
                    board_conflict = current_board != expected_board_words
                self._connection.execute(
                    """
                    INSERT INTO guesses(
                        session_id, puzzle_id, lang, word, normalized_word,
                        accepted, rejection_reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        puzzle.id,
                        puzzle.lang,
                        word,
                        None if board_conflict else normalized_word,
                        int(accepted and not board_conflict),
                        "board_mismatch" if board_conflict else rejection_reason,
                        timestamp,
                    ),
                )
                row = self._connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM guesses
                    WHERE session_id = ? AND puzzle_id = ? AND accepted = 1
                    """,
                    (session_id, puzzle.id),
                ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        if board_conflict:
            raise APIError(
                409,
                "BOARD_MISMATCH",
                "Board changed while this guess was being processed",
            )
        return int(row["count"])

    def record_solve(
        self,
        *,
        session_id: str,
        puzzle: PuzzleRecord,
        path_length: int,
        guess_count: int,
        score: float,
        solved_at: float | None = None,
    ) -> None:
        """Record a solve once, including server-measured duration."""
        current_time = solved_at if solved_at is not None else time.time()
        with self._lock:
            session = self._connection.execute(
                """
                SELECT started_at FROM sessions
                WHERE session_id = ? AND puzzle_id = ?
                """,
                (session_id, puzzle.id),
            ).fetchone()
            if session is None:
                raise APIError(
                    409,
                    "SESSION_NOT_STARTED",
                    "Load this puzzle before submitting guesses",
                )
            duration_ms = max(
                0,
                round((current_time - float(session["started_at"])) * 1000),
            )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO solves(
                    session_id, puzzle_id, lang, path_length, guess_count,
                    score, duration_ms, solved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    puzzle.id,
                    puzzle.lang,
                    path_length,
                    guess_count,
                    score,
                    duration_ms,
                    current_time,
                ),
            )

    def promote_daily_puzzle(
        self,
        language: str,
        active_date: date,
        puzzle_id: str,
        *,
        promoted_at: float | None = None,
    ) -> None:
        """Set the active daily puzzle for a language and UTC date."""
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO active_daily_puzzles(lang, active_date, puzzle_id, promoted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lang, active_date) DO UPDATE SET
                    puzzle_id = excluded.puzzle_id,
                    promoted_at = excluded.promoted_at
                """,
                (
                    language,
                    active_date.isoformat(),
                    puzzle_id,
                    promoted_at if promoted_at is not None else time.time(),
                ),
            )

    def active_daily_puzzle(self, language: str, active_date: date) -> str | None:
        """Return the rollover-selected puzzle, when one has been promoted."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT puzzle_id FROM active_daily_puzzles
                WHERE lang = ? AND active_date = ?
                """,
                (language, active_date.isoformat()),
            ).fetchone()
        return str(row["puzzle_id"]) if row else None

    def submit_feedback(
        self,
        language: str,
        responses: list[dict[str, Any]],
        *,
        created_at: float | None = None,
    ) -> None:
        """Insert a complete native-speaker batch of ten puzzle ratings."""
        if len(responses) != 10:
            raise ValueError("A playtest feedback batch must contain exactly 10 responses")
        timestamp = created_at if created_at is not None else time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.executemany(
                    """
                    INSERT INTO feedback(
                        lang, puzzle_id, rating, feedback, native_speaker, created_at
                    ) VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    [
                        (
                            language,
                            response["puzzle_id"],
                            response["rating"],
                            response["feedback"],
                            timestamp,
                        )
                        for response in responses
                    ],
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _aggregate_row(
        row: Mapping[str, Any],
        language: str | None,
    ) -> dict[str, Any]:
        return {
            "lang": language,
            "started_games": int(row["started_games"]),
            "solves": int(row["solves"]),
            "solve_rate": (
                float(row["solves"]) / int(row["started_games"])
                if row["started_games"]
                else 0.0
            ),
            "average_path_length": (
                float(row["average_path_length"])
                if row["average_path_length"] is not None
                else None
            ),
            "average_guesses": (
                float(row["average_guesses"])
                if row["average_guesses"] is not None
                else None
            ),
        }

    def _aggregate(self, language: str | None) -> dict[str, Any]:
        with self._lock:
            if language is None:
                sessions = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM sessions"
                ).fetchone()
                solves = self._connection.execute(
                    """
                    SELECT COUNT(*) AS solves, AVG(path_length) AS average_path_length,
                           AVG(guess_count) AS average_guesses
                    FROM solves
                    """
                ).fetchone()
                row = {
                    "started_games": sessions["count"],
                    **dict(solves),
                }
                return self._aggregate_row(row, None)

            sessions = self._connection.execute(
                "SELECT COUNT(*) AS count FROM sessions WHERE lang = ?",
                (language,),
            ).fetchone()
            solves = self._connection.execute(
                """
                SELECT COUNT(*) AS solves, AVG(path_length) AS average_path_length,
                       AVG(guess_count) AS average_guesses
                FROM solves WHERE lang = ?
                """,
                (language,),
            ).fetchone()
            row = {
                "started_games": sessions["count"],
                **dict(solves),
            }
        return self._aggregate_row(row, language)

    def stats(
        self,
        languages: tuple[str, ...],
        *,
        today: date,
        leaderboard_language: str | None = None,
        slow_guess_threshold: float = 35.0,
        fastest_limit: int = 10,
    ) -> dict[str, Any]:
        """Return global and per-language aggregates and today's fastest solves."""
        global_stats = self._aggregate(None)
        language_stats = [self._aggregate(language) for language in languages]
        where_language = ""
        parameters: list[object] = [today.isoformat()]
        if leaderboard_language is not None:
            where_language = " AND s.lang = ?"
            parameters.append(leaderboard_language)
        parameters.append(fastest_limit)
        with self._lock:
            fastest = self._connection.execute(
                f"""
                SELECT s.lang, s.path_length, s.guess_count, s.duration_ms, s.solved_at
                FROM solves AS s
                JOIN sessions AS p USING (session_id, puzzle_id)
                WHERE p.mode = 'daily'
                    AND date(s.solved_at, 'unixepoch') = ? {where_language}
                ORDER BY s.duration_ms ASC, s.solved_at ASC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return {
            "global": global_stats,
            "languages": language_stats,
            "today_fastest_solves": [
                {
                    "place": index,
                    "lang": row["lang"],
                    "path_length": row["path_length"],
                    "guess_count": row["guess_count"],
                    "duration_ms": row["duration_ms"],
                    "solved_at": float(row["solved_at"]),
                }
                for index, row in enumerate(fastest, start=1)
            ],
            "slow_guess_threshold": slow_guess_threshold,
        }

    def rejected_words_csv(self, language: str) -> str:
        """Export rejected words and reasons without exposing session IDs."""
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(("timestamp_utc", "lang", "puzzle_id", "word", "reason"))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT created_at, lang, puzzle_id, word, rejection_reason
                FROM guesses
                WHERE lang = ? AND accepted = 0
                ORDER BY created_at
                """,
                (language,),
            ).fetchall()
        for row in rows:
            writer.writerow(
                (
                    row["created_at"],
                    row["lang"],
                    row["puzzle_id"],
                    row["word"],
                    row["rejection_reason"] or "unknown",
                )
            )
        return output.getvalue()

    def slow_puzzles_csv(self, language: str, threshold: float = 35.0) -> str:
        """Export puzzles whose solved-game average reaches the guess threshold."""
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(("lang", "puzzle_id", "solve_count", "average_guesses"))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT lang, puzzle_id, COUNT(*) AS solve_count,
                       AVG(guess_count) AS average_guesses
                FROM solves
                WHERE lang = ?
                GROUP BY lang, puzzle_id
                HAVING AVG(guess_count) >= ?
                ORDER BY average_guesses DESC, puzzle_id
                """,
                (language, threshold),
            ).fetchall()
        for row in rows:
            writer.writerow(
                (
                    row["lang"],
                    row["puzzle_id"],
                    row["solve_count"],
                    round(float(row["average_guesses"]), 2),
                )
            )
        return output.getvalue()

    def slow_puzzles(self, threshold: float = 35.0) -> list[dict[str, Any]]:
        """Return aggregate slow-puzzle records across all languages."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT lang, puzzle_id, COUNT(*) AS solve_count,
                       AVG(guess_count) AS average_guesses
                FROM solves
                GROUP BY lang, puzzle_id
                HAVING AVG(guess_count) >= ?
                ORDER BY average_guesses DESC, lang, puzzle_id
                """,
                (threshold,),
            ).fetchall()
        return [
            {
                "lang": row["lang"],
                "puzzle_id": row["puzzle_id"],
                "solve_count": int(row["solve_count"]),
                "average_guesses": float(row["average_guesses"]),
            }
            for row in rows
        ]

    def rejected_words(self, language: str | None = None) -> list[dict[str, Any]]:
        """Return rejected-word records for internal reports."""
        query = """
            SELECT created_at, lang, puzzle_id, word, rejection_reason
            FROM guesses WHERE accepted = 0
        """
        parameters: tuple[object, ...] = ()
        if language is not None:
            query += " AND lang = ?"
            parameters = (language,)
        query += " ORDER BY created_at"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [
            {
                "created_at": float(row["created_at"]),
                "lang": row["lang"],
                "puzzle_id": row["puzzle_id"],
                "word": row["word"],
                "reason": row["rejection_reason"] or "unknown",
            }
            for row in rows
        ]
