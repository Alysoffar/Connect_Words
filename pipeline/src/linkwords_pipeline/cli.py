"""Command-line entry point for the LinkWords offline pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date
from typing import Sequence

from .calibrate import (
    ALLOWED_PERCENTILES,
    DEFAULT_PERCENTILE,
    DEFAULT_SAMPLE_COUNT,
    calibrate_languages,
)
from .config import LANGUAGE_CODES, REPOSITORY_ROOT
from .downloads import (
    FILTERS_ROOT,
    download_data,
    load_supplemental_lemmas,
    load_profanities,
    load_stopwords,
    require_vector_sources,
)
from .lexical import DATABASE_PATH, build_wordnet_database, load_wordnet_lemmas
from .generate_puzzles import generate_languages, utc_today
from .vocabulary import build_vocabulary, load_word_list
from .vectors import build_vector_store

VOCABULARY_ROOT = REPOSITORY_ROOT / "data" / "vocab"


def _languages(value: str) -> tuple[str, ...]:
    if value == "all":
        return LANGUAGE_CODES
    if value not in LANGUAGE_CODES:
        raise argparse.ArgumentTypeError(
            f"language must be one of {', '.join(LANGUAGE_CODES)} or 'all'"
        )
    return (value,)


def _build_one_language(
    language: str,
    *,
    step: str,
    force: bool,
    device: str | None,
    batch_size: int,
) -> None:
    vocabulary_path = VOCABULARY_ROOT / f"{language}.json"

    if step in {"all", "vocab"}:
        stopwords = load_stopwords(language)
        profanities = load_profanities(language)
        proper_noun_path = FILTERS_ROOT / "proper_nouns" / f"{language}.txt"
        proper_nouns = load_word_list(proper_noun_path)
        if not proper_noun_path.is_file():
            warnings.warn(
                f"No supplementary proper-noun deny list at {proper_noun_path}; "
                "configured lexical sources still filter candidates; "
                "capitalization exclusion is enabled only for configured languages.",
                stacklevel=2,
            )
        words = build_vocabulary(
            language,
            output_path=vocabulary_path,
            stopwords=stopwords,
            profanities=profanities,
            proper_nouns=proper_nouns,
            known_lemmas=(
                load_wordnet_lemmas(language)
                | load_supplemental_lemmas(language)
            ),
        )
        print(f"{language}: vocabulary ready ({len(words):,} words)")
    elif not vocabulary_path.is_file():
        raise FileNotFoundError(
            f"Missing {vocabulary_path}; run `python -m pipeline build "
            f"--lang {language} --step vocab` first"
        )

    if step in {"all", "vectors"}:
        numberbatch_path, model_path = require_vector_sources()
        output = build_vector_store(
            language,
            vocabulary_path=vocabulary_path,
            numberbatch_path=numberbatch_path,
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            force=force,
        )
        print(f"{language}: vectors ready ({output})")

    if step in {"all", "wordnet"}:
        words = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        count = build_wordnet_database(
            language,
            database_path=DATABASE_PATH,
            vocabulary=words,
            force=force,
        )
        print(f"{language}: lexical database ready ({count:,} pairs)")


def build_command(
    languages: tuple[str, ...],
    *,
    step: str = "all",
    force: bool = False,
    device: str | None = None,
    batch_size: int = 64,
) -> None:
    """Run selected offline pipeline stages for one or more languages."""
    for language in languages:
        _build_one_language(
            language,
            step=step,
            force=force,
            device=device,
            batch_size=batch_size,
        )


def create_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Build LinkWords offline language data and vector assets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download",
        help="download pinned datasets and model files (large downloads)",
    )
    download.add_argument("--lang", required=True, type=_languages)

    build = commands.add_parser("build", help="build offline artifacts from cached sources")
    build.add_argument("--lang", required=True, type=_languages)
    build.add_argument(
        "--step",
        choices=("all", "vocab", "vectors", "wordnet"),
        default="all",
    )
    build.add_argument("--force", action="store_true", help="rebuild cached artifacts")
    build.add_argument("--device", help="SentenceTransformer device, e.g. cpu or cuda")
    build.add_argument("--batch-size", type=int, default=64)

    calibrate = commands.add_parser(
        "calibrate",
        help="sample score distributions and update language thresholds",
    )
    calibrate.add_argument("--lang", required=True, type=_languages)
    calibrate.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    calibrate.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    calibrate.add_argument("--seed", type=int, default=42)

    puzzles = commands.add_parser(
        "puzzles",
        help="generate dated daily queues and practice pools",
    )
    puzzles.add_argument("--lang", required=True, type=_languages)
    puzzles.add_argument("--daily", type=int, default=30)
    puzzles.add_argument("--practice", type=int, default=500)
    puzzles.add_argument("--seed", type=int, default=42)
    puzzles.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=utc_today(),
        help="first daily puzzle date in YYYY-MM-DD format (defaults to UTC today)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline CLI and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = create_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "download":
            download_data(arguments.lang)
        elif arguments.command == "calibrate":
            if arguments.samples <= 0:
                parser.error("--samples must be positive")
            if not ALLOWED_PERCENTILES[0] <= arguments.percentile <= ALLOWED_PERCENTILES[1]:
                parser.error("--percentile must be between 98 and 98.5")
            calibrate_languages(
                arguments.lang,
                sample_count=arguments.samples,
                percentile=arguments.percentile,
                seed=arguments.seed,
            )
        elif arguments.command == "puzzles":
            if arguments.daily < 0 or arguments.practice < 0:
                parser.error("--daily and --practice cannot be negative")
            generate_languages(
                arguments.lang,
                daily_count=arguments.daily,
                practice_count=arguments.practice,
                seed=arguments.seed,
                start_date=arguments.start_date,
            )
        else:
            if arguments.batch_size <= 0:
                parser.error("--batch-size must be positive")
            build_command(
                arguments.lang,
                step=arguments.step,
                force=arguments.force,
                device=arguments.device,
                batch_size=arguments.batch_size,
            )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        print(f"pipeline: error: {error}", file=sys.stderr)
        return 1
    return 0
