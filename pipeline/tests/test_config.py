"""Tests for the shared language configuration loader."""

import json
from pathlib import Path

import pytest

from linkwords_pipeline.config import LANGUAGE_CODES, load_language_config


def test_shared_config_has_each_supported_language() -> None:
    config = load_language_config()

    assert set(config) == set(LANGUAGE_CODES)
    assert config["ar"]["rtl"] is True
    assert config["ja"]["cosine_blend_weights"] == {
        "numberbatch": 0.25,
        "bge_m3": 0.75,
    }


def test_config_rejects_missing_language(tmp_path: Path) -> None:
    config = {"en": {}}
    path = tmp_path / "languages.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected language settings"):
        load_language_config(path)
