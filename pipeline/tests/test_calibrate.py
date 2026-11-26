"""Tests for calibration sampling, summaries, and threshold persistence."""

import json
from pathlib import Path

import pytest

from linkwords_pipeline.calibrate import (
    CalibrationStats,
    calibrate_language,
    plot_distribution,
    sample_word_pairs,
    summarize_scores,
    update_thresholds,
)


def test_pair_sampling_is_unique_unordered_and_reproducible() -> None:
    words = ["a", "b", "c", "d", "e"]

    sample = sample_word_pairs(words, 8, seed=13)

    assert sample == sample_word_pairs(words, 8, seed=13)
    assert len(sample) == len({tuple(sorted(pair)) for pair in sample})
    assert all(first != second for first, second in sample)


def test_pair_sampling_rejects_impossible_request() -> None:
    with pytest.raises(ValueError, match="Cannot sample"):
        sample_word_pairs(["only", "two"], 2)


def test_summary_uses_requested_percentile_and_reports_distribution() -> None:
    summary = summarize_scores("en", [0.0, 0.1, 0.2, 0.3, 0.4], 98.25)

    assert summary["sample_count"] == 5
    assert summary["mean"] == pytest.approx(0.2)
    assert summary["std"] == pytest.approx(0.1414213562)
    assert summary["threshold"] == summary["percentiles"]["p98.25"]


def test_plot_distribution_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "plots" / "en.png"

    plot_distribution("en", [0.1, 0.2, 0.3], 0.25, output_path)

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_calibrate_language_saves_stats_and_calls_scorer_for_each_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabulary_path = tmp_path / "vocab.json"
    vocabulary_path.write_text(
        json.dumps(["worda", "wordb", "wordc", "wordd", "worde"]),
        encoding="utf-8",
    )
    calls: list[tuple[str, str, str]] = []
    import linkwords_pipeline.calibrate as calibrate_module

    def fake_plot(language, scores, threshold, output_path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")

    monkeypatch.setattr(
        calibrate_module,
        "plot_distribution",
        fake_plot,
    )

    def scorer(language: str, word_a: str, word_b: str) -> float:
        calls.append((language, word_a, word_b))
        return (ord(word_a[-1]) + ord(word_b[-1])) / 1000

    result = calibrate_language(
        "en",
        sample_count=7,
        percentile=98.5,
        seed=7,
        vocabulary_path=vocabulary_path,
        output_root=tmp_path / "calibration",
        scorer=scorer,
    )

    assert result.sample_count == 7
    assert len(calls) == 7
    assert Path(result.plot_path).read_bytes() == b"png"
    stats = json.loads(Path(result.stats_path).read_text(encoding="utf-8"))
    assert stats["threshold_percentile"] == 98.5
    assert stats["sample_count"] == 7


def test_threshold_updates_are_language_scoped(tmp_path: Path) -> None:
    config_path = tmp_path / "languages.json"
    config_path.write_text(
        json.dumps({"en": {"threshold": None}, "de": {"threshold": None}}),
        encoding="utf-8",
    )
    result = CalibrationStats(
        lang="en",
        sample_count=20_000,
        mean=0.1,
        std=0.2,
        percentiles={"p98": 0.5},
        threshold_percentile=98.25,
        threshold=0.51,
        plot_path="en.png",
        stats_path="en.json",
    )

    update_thresholds([result], config_path=config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["en"]["threshold"] == 0.51
    assert config["en"]["threshold_percentile"] == 98.25
    assert config["de"]["threshold"] is None
