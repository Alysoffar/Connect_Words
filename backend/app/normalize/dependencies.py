"""Shared configuration and data-path helpers for the normalization package."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LANGUAGE_CONFIG_PATH = REPOSITORY_ROOT / "shared" / "languages.json"


@lru_cache(maxsize=1)
def load_language_settings() -> dict[str, dict[str, Any]]:
    """Load word-length and language settings from the shared JSON config."""
    with LANGUAGE_CONFIG_PATH.open(encoding="utf-8") as config_file:
        settings: dict[str, dict[str, Any]] = json.load(config_file)
    return settings


def repository_nltk_path() -> Path:
    """Return the shared local NLTK corpus directory."""
    return REPOSITORY_ROOT / "data" / "nltk_data"
