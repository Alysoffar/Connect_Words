"""Tests for the pipeline CLI argument parser."""

import pytest

from linkwords_pipeline.cli import create_parser


def test_cli_accepts_all_languages_and_staged_build() -> None:
    args = create_parser().parse_args(
        ["build", "--lang", "all", "--step", "vectors", "--device", "cpu"]
    )

    assert args.lang == ("en", "de", "es", "ar", "ja")
    assert args.step == "vectors"
    assert args.device == "cpu"


def test_cli_rejects_unsupported_language() -> None:
    with pytest.raises(SystemExit):
        create_parser().parse_args(["build", "--lang", "fr"])


def test_cli_accepts_calibration_and_puzzle_options() -> None:
    parser = create_parser()

    calibration = parser.parse_args(
        ["calibrate", "--lang", "ar", "--samples", "20000", "--percentile", "98.5"]
    )
    puzzles = parser.parse_args(
        ["puzzles", "--lang", "ar", "--daily", "30", "--practice", "500"]
    )

    assert calibration.lang == ("ar",)
    assert calibration.percentile == 98.5
    assert puzzles.daily == 30
    assert puzzles.practice == 500
