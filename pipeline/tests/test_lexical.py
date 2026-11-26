"""Tests for building WordNet relation rows without external corpus downloads."""

import sqlite3
import sys
import types
from pathlib import Path

import pytest

from linkwords_pipeline.lexical import (
    _matching_wordnet_languages,
    build_wordnet_database,
    lexical_boost_score,
    load_wordnet_reader,
)


class _Synset:
    def __init__(self, words: list[str]) -> None:
        self.words = words
        self.parents: list[_Synset] = []
        self.children: list[_Synset] = []

    def lemma_names(self, lang: str) -> list[str]:
        assert lang == "eng"
        return self.words

    def hypernyms(self) -> list["_Synset"]:
        return self.parents

    def hyponyms(self) -> list["_Synset"]:
        return self.children


class _FakeWordNet:
    def __init__(self, synsets: list[_Synset]) -> None:
        self.synsets = synsets
        self.language_codes = ["eng"]

    def ensure_loaded(self) -> None:
        return None

    def langs(self) -> list[str]:
        return self.language_codes

    def add_exomw(self) -> None:
        self.language_codes.extend(["deu_cldr", "deu_wikt"])

    def all_synsets(self) -> list[_Synset]:
        return self.synsets


class _FakeNltk:
    data = types.SimpleNamespace(path=[])


def test_extended_omw_language_aliases_are_selected_for_german() -> None:
    assert _matching_wordnet_languages(
        "de",
        ("eng", "deu_cldr", "deu_wikt", "spa"),
    ) == ("deu_cldr", "deu_wikt")


def test_wordnet_reader_loads_extended_omw(monkeypatch: pytest.MonkeyPatch) -> None:
    wordnet = _FakeWordNet([])
    corpus = types.ModuleType("nltk.corpus")
    corpus.wordnet = wordnet
    nltk_module = types.ModuleType("nltk")
    nltk_module.data = _FakeNltk.data
    monkeypatch.setitem(sys.modules, "nltk", nltk_module)
    monkeypatch.setitem(sys.modules, "nltk.corpus", corpus)

    reader, language_codes = load_wordnet_reader("de")

    assert reader is wordnet
    assert language_codes == ("deu_cldr", "deu_wikt")


def test_build_wordnet_emits_synonym_and_taxonomic_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dog = _Synset(["dog", "canine"])
    animal = _Synset(["animal"])
    dog.parents.append(animal)
    animal.children.append(dog)
    corpus = types.ModuleType("nltk.corpus")
    corpus.wordnet = _FakeWordNet([dog, animal])
    nltk_module = types.ModuleType("nltk")
    nltk_module.data = _FakeNltk.data
    monkeypatch.setitem(sys.modules, "nltk", nltk_module)
    monkeypatch.setitem(sys.modules, "nltk.corpus", corpus)

    database_path = tmp_path / "linkwords.sqlite"
    pair_count = build_wordnet_database(
        "en",
        database_path=database_path,
        vocabulary={"dog", "canine", "animal"},
        force=True,
    )

    assert pair_count >= 2
    assert lexical_boost_score("en", "dog", "canine", database_path) >= 0.75
    assert lexical_boost_score("en", "dog", "animal", database_path) >= 0.75
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM synonyms").fetchone()[0] == pair_count
