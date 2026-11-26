"""Calibrate per-language connection thresholds from random vocabulary pairs."""

from __future__ import annotations

import json
import os
import random
import tempfile
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import LANGUAGE_CODES, REPOSITORY_ROOT
from .scoring import score

DEFAULT_SAMPLE_COUNT = 20_000
DEFAULT_PERCENTILE = 98.25
ALLOWED_PERCENTILES = (98.0, 98.5)
VOCABULARY_ROOT = REPOSITORY_ROOT / "data" / "vocab"
CALIBRATION_ROOT = REPOSITORY_ROOT / "data" / "calibration"
SHARED_CONFIG = REPOSITORY_ROOT / "shared" / "languages.json"
ScoreFunction = Callable[[str, str, str], float]


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    """Distribution summary and selected threshold for one language."""

    lang: str
    sample_count: int
    mean: float
    std: float
    percentiles: dict[str, float]
    threshold_percentile: float
    threshold: float
    plot_path: str
    stats_path: str


def sample_word_pairs(
    words: Sequence[str],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    *,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Sample unique unordered pairs uniformly without replacement."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if len(set(words)) != len(words):
        raise ValueError("Vocabulary words must be unique")
    word_count = len(words)
    possible_pairs = word_count * (word_count - 1) // 2
    if sample_count > possible_pairs:
        raise ValueError(
            f"Cannot sample {sample_count} unique pairs from {word_count} words"
        )

    row_starts = [
        row * (2 * word_count - row - 1) // 2
        for row in range(word_count - 1)
    ]
    pair_indices = random.Random(seed).sample(range(possible_pairs), sample_count)
    pairs: list[tuple[str, str]] = []
    for pair_index in pair_indices:
        row = bisect_right(row_starts, pair_index) - 1
        column = row + 1 + pair_index - row_starts[row]
        pairs.append((words[row], words[column]))
    return pairs


def summarize_scores(
    language: str,
    values: Sequence[float],
    percentile: float,
) -> dict[str, Any]:
    """Calculate distribution statistics and the configured percentile cutoff."""
    if not values:
        raise ValueError("At least one score is required")
    if not ALLOWED_PERCENTILES[0] <= percentile <= ALLOWED_PERCENTILES[1]:
        raise ValueError("percentile must be between 98 and 98.5")
    scores = np.asarray(values, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values")
    requested_percentiles = sorted(
        {50.0, 90.0, 95.0, 98.0, 98.25, 98.5, 99.0, percentile}
    )
    percentiles = {
        f"p{value:g}": float(np.percentile(scores, value))
        for value in requested_percentiles
    }
    return {
        "lang": language,
        "sample_count": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "percentiles": percentiles,
        "threshold_percentile": percentile,
        "threshold": percentiles[f"p{percentile:g}"],
    }


def plot_distribution(
    language: str,
    scores: Sequence[float],
    threshold: float,
    output_path: Path,
) -> None:
    """Save a score histogram with the selected threshold marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(scores, bins=60, color="#4472C4", alpha=0.85)
    axis.axvline(
        threshold,
        color="#C00000",
        linestyle="--",
        label=f"threshold {threshold:.4f}",
    )
    axis.set(
        title=f"LinkWords pair similarity distribution ({language})",
        xlabel="score(lang, word_a, word_b)",
        ylabel="pair count",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def calibrate_language(
    language: str,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    percentile: float = DEFAULT_PERCENTILE,
    seed: int = 42,
    vocabulary_path: Path | None = None,
    output_root: Path = CALIBRATION_ROOT,
    scorer: ScoreFunction = score,
) -> CalibrationStats:
    """Sample and score unique vocabulary pairs for one language."""
    if language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language: {language}")
    path = vocabulary_path or VOCABULARY_ROOT / f"{language}.json"
    words: list[str] = json.loads(path.read_text(encoding="utf-8"))
    pairs = sample_word_pairs(words, sample_count, seed=seed)
    values = [
        float(scorer(language, word_a, word_b))
        for word_a, word_b in pairs
    ]
    summary = summarize_scores(language, values, percentile)
    plot_path = output_root / f"{language}.png"
    stats_path = output_root / f"{language}.json"
    plot_distribution(language, values, summary["threshold"], plot_path)
    _atomic_write_json(
        stats_path,
        {
            **summary,
            "seed": seed,
            "vocabulary_path": str(path),
            "plot_path": str(plot_path),
        },
    )
    return CalibrationStats(
        lang=language,
        sample_count=summary["sample_count"],
        mean=summary["mean"],
        std=summary["std"],
        percentiles=summary["percentiles"],
        threshold_percentile=percentile,
        threshold=summary["threshold"],
        plot_path=str(plot_path),
        stats_path=str(stats_path),
    )


def update_thresholds(
    results: Sequence[CalibrationStats],
    *,
    config_path: Path = SHARED_CONFIG,
) -> None:
    """Atomically update thresholds and their selected percentile in shared config."""
    config: dict[str, dict[str, Any]] = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    for result in results:
        config[result.lang]["threshold"] = result.threshold
        config[result.lang]["threshold_percentile"] = result.threshold_percentile
    temporary_path = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, config_path)


def print_comparison_table(results: Sequence[CalibrationStats]) -> None:
    """Print a compact, aligned calibration comparison."""
    print(
        f"{'lang':<6} {'n':>8} {'mean':>10} {'std':>10} "
        f"{'p98':>10} {'threshold p':>12} {'threshold':>10}"
    )
    for result in results:
        print(
            f"{result.lang:<6} {result.sample_count:>8} "
            f"{result.mean:>10.4f} {result.std:>10.4f} "
            f"{result.percentiles['p98']:>10.4f} "
            f"{result.threshold_percentile:>11.2f}% {result.threshold:>10.4f}"
        )


def calibrate_languages(
    languages: Sequence[str],
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    percentile: float = DEFAULT_PERCENTILE,
    seed: int = 42,
    vocabulary_root: Path = VOCABULARY_ROOT,
    output_root: Path = CALIBRATION_ROOT,
    config_path: Path = SHARED_CONFIG,
    scorer: ScoreFunction = score,
) -> list[CalibrationStats]:
    """Calibrate selected languages, save artifacts, and update shared thresholds."""
    results = [
        calibrate_language(
            language,
            sample_count=sample_count,
            percentile=percentile,
            seed=seed + LANGUAGE_CODES.index(language),
            vocabulary_path=vocabulary_root / f"{language}.json",
            output_root=output_root,
            scorer=scorer,
        )
        for language in languages
    ]
    update_thresholds(results, config_path=config_path)
    print_comparison_table(results)
    return results
