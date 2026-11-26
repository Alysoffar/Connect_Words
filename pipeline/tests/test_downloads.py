"""Tests for resumable data source downloads."""

import json
import sys
import types
from pathlib import Path

import pytest

from linkwords_pipeline import downloads


class _Response:
    status = 206
    headers = {"Content-Length": "4"}

    def __init__(self) -> None:
        self._data = b"data"

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk


def test_download_resumes_partial_file_and_promotes_it_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "source.dat"
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(b"part")
    requests = []

    def fake_urlopen(request):
        requests.append(request)
        return _Response()

    monkeypatch.setattr(downloads.urllib.request, "urlopen", fake_urlopen)
    downloads.download_file("https://example.test/source", destination)

    assert requests[0].get_header("Range") == "bytes=4-"
    assert destination.read_bytes() == b"partdata"
    assert not partial.exists()


def test_download_skips_unused_onnx_model_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filters_root = tmp_path / "filters"
    monkeypatch.setattr(downloads, "SOURCES_ROOT", tmp_path / "sources")
    monkeypatch.setattr(downloads, "FILTERS_ROOT", filters_root)
    monkeypatch.setattr(downloads, "NLTK_DATA_ROOT", tmp_path / "nltk_data")
    monkeypatch.setattr(downloads, "MODEL_ROOT", tmp_path / "model")
    downloaded_urls: list[str] = []

    def fake_download_file(url: str, destination: Path) -> None:
        downloaded_urls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "stopwords-iso.json":
            destination.write_text(json.dumps({"de": []}), encoding="utf-8")
        else:
            destination.write_text("word\n", encoding="utf-8")

    snapshot_arguments: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> None:
        snapshot_arguments.update(kwargs)

    nltk_module = types.ModuleType("nltk")
    nltk_module.data = types.SimpleNamespace(path=[])
    nltk_module.download = lambda *_args, **_kwargs: True
    huggingface_module = types.ModuleType("huggingface_hub")
    huggingface_module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "nltk", nltk_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_module)
    monkeypatch.setattr(downloads, "download_file", fake_download_file)

    downloads.download_data(("de",))

    assert snapshot_arguments["ignore_patterns"] == ["onnx/**"]
    assert downloads.LEMMA_LIST_URLS["de"] in downloaded_urls


def test_supplemental_lemma_loader_reads_hunspell_headwords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources_root = tmp_path / "sources"
    lemma_path = sources_root / "lemmas" / "de.dic"
    lemma_path.parent.mkdir(parents=True)
    lemma_path.write_text(
        "3\nKrankenhaus/Nm\nÄrztin/Fm\n# comment\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(downloads, "SOURCES_ROOT", sources_root)

    assert downloads.load_supplemental_lemmas("de") == {
        "Krankenhaus",
        "Ärztin",
    }
