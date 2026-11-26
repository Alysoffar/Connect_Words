"""Tests for UTC daily queue promotion and low-queue alerts."""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.analytics import AnalyticsStore
from app.rollover import promote_daily_queues, seconds_until_next_utc_midnight


def _collection(path: Path, language: str, dates: list[date]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    puzzles = [
        {
            "id": f"{language}-daily-{puzzle_date.isoformat()}",
            "lang": language,
            "anchorA": "start",
            "anchorB": "finish",
            "shortestPath": ["start", "finish"],
            "alternativePaths": [],
            "difficulty": "easy",
            "date": puzzle_date.isoformat(),
        }
        for puzzle_date in dates
    ]
    path.write_text(
        json.dumps(
            {
                "lang": language,
                "daily_queue": puzzles,
                "practice_pool": [],
            }
        ),
        encoding="utf-8",
    )


def test_rollover_promotes_today_and_warns_below_seven_days(tmp_path: Path) -> None:
    today = date(2026, 11, 26)
    _collection(
        tmp_path / "puzzles" / "en.json",
        "en",
        [today + timedelta(days=offset) for offset in range(6)],
    )
    alerts: list[dict[str, object]] = []

    with AnalyticsStore(":memory:") as analytics:
        report = promote_daily_queues(
            data_root=tmp_path,
            analytics=analytics,
            languages=("en",),
            active_date=today,
            alert_sender=alerts.append,
        )
        active = analytics.active_daily_puzzle("en", today)

    assert report.promoted == {"en": "en-daily-2026-11-26"}
    assert report.queued_days == {"en": 6}
    assert report.low_queue == {"en": 6}
    assert active == "en-daily-2026-11-26"
    assert alerts[0]["lowQueueDays"] == {"en": 6}


def test_rollover_does_not_promote_a_future_puzzle_as_today(tmp_path: Path) -> None:
    today = date(2026, 11, 26)
    _collection(
        tmp_path / "puzzles" / "de.json",
        "de",
        [today + timedelta(days=offset) for offset in range(1, 9)],
    )

    with AnalyticsStore(":memory:") as analytics:
        report = promote_daily_queues(
            data_root=tmp_path,
            analytics=analytics,
            languages=("de",),
            active_date=today,
        )
        active = analytics.active_daily_puzzle("de", today)

    assert report.queued_days == {"de": 8}
    assert report.missing_today == ("de",)
    assert active is None


def test_midnight_delay_is_calculated_in_utc() -> None:
    local_time = datetime(
        2026,
        11,
        26,
        23,
        30,
        tzinfo=timezone(timedelta(hours=2)),
    )

    assert seconds_until_next_utc_midnight(local_time) == 9_000


def test_duplicate_queued_dates_are_rejected(tmp_path: Path) -> None:
    today = date(2026, 11, 26)
    _collection(
        tmp_path / "puzzles" / "es.json",
        "es",
        [today, today],
    )

    with AnalyticsStore(":memory:") as analytics:
        with pytest.raises(ValueError, match="Duplicate daily puzzle date"):
            promote_daily_queues(
                data_root=tmp_path,
                analytics=analytics,
                languages=("es",),
                active_date=today,
            )
