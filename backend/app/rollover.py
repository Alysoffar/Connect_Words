"""UTC daily-puzzle queue promotion and low-queue alerting."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .analytics import AnalyticsStore
from .normalize.dependencies import REPOSITORY_ROOT, load_language_settings
from .schemas import PuzzleCollection

LOGGER = logging.getLogger("linkwords.rollover")
DEFAULT_MINIMUM_QUEUE_DAYS = 7


@dataclass(frozen=True, slots=True)
class RolloverReport:
    """Outcome of one UTC rollover pass."""

    active_date: date
    promoted: Mapping[str, str]
    queued_days: Mapping[str, int]
    low_queue: Mapping[str, int]
    missing_today: tuple[str, ...]


def seconds_until_next_utc_midnight(now: datetime) -> float:
    """Return seconds until the next midnight in UTC."""
    current = now.astimezone(timezone.utc)
    next_day = current.date() + timedelta(days=1)
    next_midnight = datetime.combine(
        next_day,
        datetime_time.min,
        tzinfo=timezone.utc,
    )
    return max(0.0, (next_midnight - current).total_seconds())


def _read_collection(path: Path, language: str) -> PuzzleCollection:
    if not path.is_file():
        raise FileNotFoundError(f"Missing puzzle collection for {language}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid puzzle collection JSON: {path}") from error
    collection = PuzzleCollection.model_validate(payload)
    if collection.lang != language:
        raise ValueError(f"Puzzle collection language mismatch: {path}")
    return collection


def promote_daily_queues(
    *,
    data_root: Path,
    analytics: AnalyticsStore,
    languages: tuple[str, ...],
    active_date: date,
    minimum_queue_days: int = DEFAULT_MINIMUM_QUEUE_DAYS,
    alert_sender: Callable[[Mapping[str, Any]], None] | None = None,
) -> RolloverReport:
    """Promote each language's dated puzzle and report queues under seven days."""
    if minimum_queue_days < 1:
        raise ValueError("minimum_queue_days must be positive")

    promoted: dict[str, str] = {}
    queued_days: dict[str, int] = {}
    low_queue: dict[str, int] = {}
    missing_today: list[str] = []

    for language in languages:
        collection = _read_collection(
            data_root / "puzzles" / f"{language}.json",
            language,
        )
        dated_puzzles: dict[date, str] = {}
        for puzzle in collection.daily_queue:
            if puzzle.date is None:
                continue
            puzzle_date = date.fromisoformat(puzzle.date)
            if puzzle_date in dated_puzzles:
                raise ValueError(
                    f"Duplicate daily puzzle date for {language}: {puzzle_date}"
                )
            dated_puzzles[puzzle_date] = puzzle.id

        days_remaining = sum(
            puzzle_date >= active_date for puzzle_date in dated_puzzles
        )
        queued_days[language] = days_remaining
        if days_remaining < minimum_queue_days:
            low_queue[language] = days_remaining

        puzzle_id = dated_puzzles.get(active_date)
        if puzzle_id is None:
            missing_today.append(language)
            continue
        analytics.promote_daily_puzzle(language, active_date, puzzle_id)
        promoted[language] = puzzle_id

    report = RolloverReport(
        active_date=active_date,
        promoted=promoted,
        queued_days=queued_days,
        low_queue=low_queue,
        missing_today=tuple(missing_today),
    )
    if report.low_queue or report.missing_today:
        LOGGER.error(
            "Daily puzzle queue alert: low_queue=%s missing_today=%s date=%s",
            dict(report.low_queue),
            report.missing_today,
            active_date.isoformat(),
        )
        if alert_sender is not None:
            alert_sender(
                {
                    "event": "linkwords_daily_queue_alert",
                    "date": active_date.isoformat(),
                    "lowQueueDays": dict(report.low_queue),
                    "missingToday": list(report.missing_today),
                }
            )
    else:
        LOGGER.info(
            "Promoted daily puzzles for %s: %s",
            active_date.isoformat(),
            dict(report.promoted),
        )
    return report


def send_webhook_alert(url: str, payload: Mapping[str, Any]) -> None:
    """Post a queue alert to an operator-configured webhook."""
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"Queue alert webhook returned HTTP {response.status}"
                )
    except (OSError, URLError, TimeoutError):
        LOGGER.exception("Could not deliver daily queue alert webhook")


def _run_once(today: date | None = None) -> RolloverReport:
    settings = load_language_settings()
    data_root = Path(os.environ.get("LINKWORDS_DATA_ROOT", REPOSITORY_ROOT / "data"))
    analytics_path = Path(
        os.environ.get(
            "LINKWORDS_ANALYTICS_DB",
            data_root / "analytics.sqlite",
        )
    )
    webhook_url = os.environ.get("LINKWORDS_ALERT_WEBHOOK_URL", "")
    with AnalyticsStore(analytics_path) as analytics:
        return promote_daily_queues(
            data_root=data_root,
            analytics=analytics,
            languages=tuple(settings),
            active_date=today or datetime.now(timezone.utc).date(),
            alert_sender=(
                (lambda payload: send_webhook_alert(webhook_url, payload))
                if webhook_url
                else None
            ),
        )


def watch_daily_rollovers(
    *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run a startup promotion and then promote once at every UTC midnight."""
    while True:
        current = now().astimezone(timezone.utc)
        _run_once(current.date())
        sleep(seconds_until_next_utc_midnight(current))


def main(argv: list[str] | None = None) -> int:
    """Run one rollover pass or stay alive as a UTC-scheduled container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="promote today's queues once, then exit",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="override today's UTC date for operations or tests",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    if arguments.once:
        report = _run_once(arguments.date)
        return 2 if report.low_queue or report.missing_today else 0
    if arguments.date is not None:
        parser.error("--date can only be used with --once")
    watch_daily_rollovers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
