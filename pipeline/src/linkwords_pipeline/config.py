"""Load and validate the repository's shared language configuration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LANGUAGE_CONFIG_PATH = REPOSITORY_ROOT / "shared" / "languages.json"


def _read_configured_language_codes() -> tuple[str, ...]:
    with LANGUAGE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        return tuple(json.load(config_file))


LANGUAGE_CODES = _read_configured_language_codes()


@lru_cache(maxsize=1)
def load_language_config(path: Path = LANGUAGE_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Load language settings and reject incomplete or invalid configuration."""
    with path.open(encoding="utf-8") as config_file:
        config: dict[str, dict[str, Any]] = json.load(config_file)

    if set(config) != set(LANGUAGE_CODES):
        raise ValueError(f"Expected language settings for {', '.join(LANGUAGE_CODES)}")

    required = {
        "min_word_length",
        "max_word_length",
        "threshold",
        "rtl",
        "font",
        "exclude_capitalized_candidates",
        "cosine_blend_weights",
    }
    for language, settings in config.items():
        missing = required - settings.keys()
        if missing:
            raise ValueError(f"{language} is missing settings: {', '.join(sorted(missing))}")
        weights = settings["cosine_blend_weights"]
        if set(weights) != {"numberbatch", "bge_m3"}:
            raise ValueError(f"{language} must configure Numberbatch and BGE-M3 weights")
        if any(not isinstance(weight, (int, float)) or weight < 0 for weight in weights.values()):
            raise ValueError(f"{language} cosine blend weights must be non-negative numbers")
        if sum(weights.values()) <= 0:
            raise ValueError(f"{language} cosine blend weights must have a positive sum")

    return config


def get_language_settings(language: str) -> dict[str, Any]:
    """Return settings for a supported language or raise a clear error."""
    config = load_language_config()
    try:
        return config[language]
    except KeyError as error:
        raise ValueError(f"Unsupported language: {language}") from error
