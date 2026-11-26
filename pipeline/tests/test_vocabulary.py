"""Tests for language-aware vocabulary filtering."""

from pathlib import Path

import pytest

from linkwords_pipeline.config import get_language_settings
from linkwords_pipeline.normalization import is_single_kanji, normalize_word
from linkwords_pipeline.vocabulary import build_vocabulary, is_clean_word


@pytest.mark.parametrize(
    ("language", "candidate", "expected"),
    [
        ("en", "Hello", "hello"),
        ("en", "O'Clock", "oclock"),
        ("ar", "ـمَرْحَبًا", "مرحبا"),
        ("ar", "أإآ ى", "ااا ي"),
        ("ja", "猫", "猫"),
        ("ja", "ｶﾀｶﾅ", "カタカナ"),
    ],
)
def test_normalization(language: str, candidate: str, expected: str) -> None:
    assert normalize_word(candidate, language) == expected


def test_clean_word_drops_urls_numbers_and_exclusion_lists() -> None:
    assert not is_clean_word("example.com", "en")
    assert not is_clean_word("word123", "en")
    assert not is_clean_word("common", "en", stopwords={"common"})
    assert not is_clean_word("badword", "en", profanities={"badword"})
    assert not is_clean_word("london", "en", proper_nouns={"London"})
    assert is_clean_word("language", "en")


def test_only_single_kanji_passes_japanese_single_character_exception() -> None:
    assert is_single_kanji("猫")
    assert is_clean_word("猫", "ja")
    assert not is_clean_word("あ", "ja")
    assert is_clean_word("猫", "ja", known_lemmas={"猫"})
    assert not is_clean_word("猫", "ja", known_lemmas={"犬"})


def test_build_vocabulary_writes_deduplicated_words(tmp_path: Path) -> None:
    words = [f"word{index}" for index in range(10)]

    def suffix(index: int) -> str:
        letters = ""
        while index:
            index, remainder = divmod(index, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters or "a"

    words.extend(f"cleanword{suffix(index)}" for index in range(30_003))
    excluded = ["cleanword" + suffix(index) for index in range(3)]
    output = tmp_path / "en.json"

    vocabulary = build_vocabulary(
        "en",
        output_path=output,
        target_size=30_000,
        candidates=words,
        stopwords={excluded[0]},
        profanities={excluded[1]},
        proper_nouns={excluded[2]},
        known_lemmas=set(words),
    )

    assert len(vocabulary) == 30_000
    assert vocabulary[0] == "cleanword" + suffix(3)
    assert output.is_file()
    assert all(word not in vocabulary for word in excluded)


def test_german_capitalized_noun_is_kept_if_it_is_an_omw_lemma(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def suffix(index: int) -> str:
        letters = ""
        while index:
            index, remainder = divmod(index, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters or "a"

    fillers = [f"cleanword{suffix(index)}" for index in range(30_000)]
    candidates = ["Krankenhaus", *fillers]
    output = tmp_path / "de.json"
    caplog.set_level("INFO", logger="linkwords_pipeline.vocabulary")

    vocabulary = build_vocabulary(
        "de",
        output_path=output,
        target_size=30_000,
        candidates=candidates,
        known_lemmas={"Krankenhaus", *fillers},
    )

    assert "krankenhaus" in vocabulary
    assert "de vocabulary rule removals:" in caplog.text
    assert "capitalization=0" in caplog.text
    assert "not_in_lexicon=0" in caplog.text


@pytest.mark.parametrize("language", ["en", "es"])
def test_english_and_spanish_capitalized_candidates_are_excluded(
    language: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def suffix(index: int) -> str:
        letters = ""
        while index:
            index, remainder = divmod(index, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters or "a"

    fillers = [f"cleanword{suffix(index)}" for index in range(30_000)]
    capitalized = "Capitalized"
    candidates = [capitalized, *fillers]
    output = tmp_path / f"{language}.json"
    caplog.set_level("INFO", logger="linkwords_pipeline.vocabulary")

    vocabulary = build_vocabulary(
        language,
        output_path=output,
        target_size=30_000,
        candidates=candidates,
        known_lemmas={capitalized.lower(), *fillers},
    )

    assert capitalized.lower() not in vocabulary
    assert "capitalization=1" in caplog.text


@pytest.mark.parametrize("language", ["de", "ar", "ja"])
def test_other_languages_skip_capitalization_exclusion(language: str) -> None:
    assert get_language_settings(language)["exclude_capitalized_candidates"] is False


def test_omw_membership_and_proper_noun_list_removals_are_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def suffix(index: int) -> str:
        letters = ""
        while index:
            index, remainder = divmod(index, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters or "a"

    fillers = [f"cleanword{suffix(index)}" for index in range(30_000)]
    candidates = ["unknownlemma", "listedname", *fillers]
    caplog.set_level("INFO", logger="linkwords_pipeline.vocabulary")

    build_vocabulary(
        "de",
        output_path=tmp_path / "de.json",
        target_size=30_000,
        candidates=candidates,
        known_lemmas={"listedname", *fillers},
        proper_nouns={"listedname"},
    )

    assert "proper_noun_list=1" in caplog.text
    assert "not_in_lexicon=1" in caplog.text
