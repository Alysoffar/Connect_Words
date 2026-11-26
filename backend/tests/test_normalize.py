"""Parametrized multilingual normalization and validation coverage."""

from pathlib import Path

import json

import pytest

from app.normalize import (
    NormalizationService,
    ValidationReason,
    normalize_and_validate,
)
from app.normalize.ar import ArabicNormalizer
from app.normalize.de import GermanNormalizer
from app.normalize.en import EnglishNormalizer
from app.normalize.es import SpanishNormalizer
from app.normalize.index import VocabularyIndex
from app.normalize.ja import JapaneseNormalizer
from app.normalize.service import configure_normalizer


@pytest.fixture
def service() -> NormalizationService:
    """Create an in-memory service with stable vocabularies and blocklists."""
    vocabularies = {
        "en": {
            "cat", "dog", "run", "play", "mouse", "cant", "blocked",
            "abcdefghijklmn", "abcdefghijklmno",
        },
        "de": {
            "krankenhaus", "straße", "hund", "haus", "donaudampfschifffahrt",
            "blocked", "abcdefghijklmnopqrstuvwxyz",
        },
        "ar": {
            "اكل", "مرحبا", "مدرسة", "فتي", "كتاب", "سلام", "ممنوع",
            "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
        },
        "es": {
            "árbol", "niño", "nino", "casa", "canción", "rápido",
            "abcdefghijklmnop", "blocked",
        },
        "ja": {
            "水", "山", "愛", "ミズ", "ABC", "blocked",
            "あいうえおかきくけこ",
        },
    }
    blocklists = {
        "en": {"blocked"},
        "de": {"blocked"},
        "ar": {"ممنوع"},
        "es": {"blocked"},
        "ja": {"blocked"},
    }
    japanese = JapaneseNormalizer()
    normalizers = {
        "en": EnglishNormalizer(),
        "de": GermanNormalizer(),
        "ar": ArabicNormalizer(lemma_provider=lambda _word: ()),
        "es": SpanishNormalizer(),
        "ja": japanese,
    }
    service = NormalizationService(
        vocabularies,
        blocklists=blocklists,
        normalizers=normalizers,
    )
    service.indexes["ja"] = VocabularyIndex(
        vocabularies["ja"],
        japanese,
        aliases={"ミズ": {"水"}},
    )
    return service


@pytest.mark.parametrize(
    ("word", "expected_normalized", "expected_reason"),
    [
        ("cat", "cat", None),
        ("CAT", "cat", None),
        ("cats", "cat", None),
        ("running", "run", None),
        ("played", "play", None),
        ("  dog  ", "dog", None),
        ("!!cat??", "cat", None),
        ("ab", "ab", ValidationReason.TOO_SHORT),
        ("abcdefghijklmnop", "abcdefghijklmnop", ValidationReason.TOO_LONG),
        ("cat123", "cat123", ValidationReason.INVALID_CHARS),
        ("unknown", "unknown", ValidationReason.NOT_IN_VOCAB),
        ("blocked", "blocked", ValidationReason.BLOCKED),
        ("", "", ValidationReason.INVALID_CHARS),
        ("can't", "cant", None),
        ("mice", "mouse", None),
    ],
)
def test_english_normalization_cases(
    service: NormalizationService,
    word: str,
    expected_normalized: str,
    expected_reason: ValidationReason | None,
) -> None:
    result = service.normalize_and_validate(word, "en")

    assert result.normalized == expected_normalized
    assert result.reason == expected_reason
    assert result.ok is (expected_reason is None)


@pytest.mark.parametrize(
    ("word", "expected_normalized", "expected_reason"),
    [
        ("Krankenhaus", "krankenhaus", None),
        ("KRANKENHAUS", "krankenhaus", None),
        ("Straße", "straße", None),
        ("STRASSE", "straße", None),
        ("Donaudampfschifffahrt", "donaudampfschifffahrt", None),
        ("ab", "ab", ValidationReason.TOO_SHORT),
        ("abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz", ValidationReason.TOO_LONG),
        ("Haus123", "haus123", ValidationReason.INVALID_CHARS),
        ("unbekannt", "unbekannt", ValidationReason.NOT_IN_VOCAB),
        ("blocked", "blocked", ValidationReason.BLOCKED),
        ("Hund", "hund", None),
        ("straße!", "straße!", ValidationReason.INVALID_CHARS),
        ("  Krankenhaus  ", "krankenhaus", None),
        ("HÄUSER", "häuser", ValidationReason.NOT_IN_VOCAB),
        ("Straße", "straße", None),
    ],
)
def test_german_normalization_cases(
    service: NormalizationService,
    word: str,
    expected_normalized: str,
    expected_reason: ValidationReason | None,
) -> None:
    result = service.normalize_and_validate(word, "de")

    assert result.normalized == expected_normalized
    assert result.reason == expected_reason
    assert result.ok is (expected_reason is None)
    if word == "Krankenhaus":
        assert result.display == "Krankenhaus"


@pytest.mark.parametrize(
    ("word", "expected_normalized", "expected_reason"),
    [
        ("أكل", "اكل", None),
        ("إكل", "اكل", None),
        ("آكل", "اكل", None),
        ("مَرْحَبًا", "مرحبا", None),
        ("مـرحبا", "مرحبا", None),
        ("مدرَسه", "مدرسة", None),
        ("فتى", "فتي", None),
        ("ا", "ا", ValidationReason.TOO_SHORT),
        ("ابتثجحخدذرزسشصضطظعغفقكلمن", "ابتثجحخدذرزسشصضطظعغفقكلمن", ValidationReason.TOO_LONG),
        ("مرحبا1", "مرحبا1", ValidationReason.INVALID_CHARS),
        ("hello", "hello", ValidationReason.INVALID_CHARS),
        ("غيرمعروف", "غيرمعروف", ValidationReason.NOT_IN_VOCAB),
        ("ممنوع", "ممنوع", ValidationReason.BLOCKED),
        ("سلام", "سلام", None),
        ("  كتاب ", "كتاب", None),
        ("مَدْرَسَة", "مدرسة", None),
    ],
)
def test_arabic_normalization_cases(
    service: NormalizationService,
    word: str,
    expected_normalized: str,
    expected_reason: ValidationReason | None,
) -> None:
    result = service.normalize_and_validate(word, "ar")

    assert result.normalized == expected_normalized
    assert result.reason == expected_reason
    assert result.ok is (expected_reason is None)


@pytest.mark.parametrize(
    ("word", "expected_normalized", "expected_reason"),
    [
        ("árbol", "árbol", None),
        ("arbol", "árbol", None),
        ("NIÑO", "niño", None),
        ("nino", "nino", None),
        ("niño", "niño", None),
        ("casa", "casa", None),
        ("canción", "canción", None),
        ("rápido", "rápido", None),
        ("ab", "ab", ValidationReason.TOO_SHORT),
        ("abcdefghijklmnopq", "abcdefghijklmnopq", ValidationReason.TOO_LONG),
        ("casa2", "casa2", ValidationReason.INVALID_CHARS),
        ("desconocido", "desconocido", ValidationReason.NOT_IN_VOCAB),
        ("blocked", "blocked", ValidationReason.BLOCKED),
        ("árbol!", "árbol!", ValidationReason.INVALID_CHARS),
        ("  ARBOL ", "árbol", None),
    ],
)
def test_spanish_normalization_cases(
    service: NormalizationService,
    word: str,
    expected_normalized: str,
    expected_reason: ValidationReason | None,
) -> None:
    result = service.normalize_and_validate(word, "es")

    assert result.normalized == expected_normalized
    assert result.reason == expected_reason
    assert result.ok is (expected_reason is None)


@pytest.mark.parametrize(
    ("word", "expected_normalized", "expected_reason"),
    [
        ("水", "水", None),
        ("山", "山", None),
        ("愛", "愛", None),
        ("猫", "猫", ValidationReason.NOT_IN_VOCAB),
        ("あ", "あ", ValidationReason.TOO_SHORT),
        ("ﾐｽﾞ", "水", None),
        ("ミズ", "水", None),
        ("ＡＢＣ", "ABC", None),
        ("あいうえおかきくけこ", "あいうえおかきくけこ", None),
        ("あいうえおかきくけこさ", "あいうえおかきくけこさ", ValidationReason.TOO_LONG),
        ("水1", "水1", ValidationReason.INVALID_CHARS),
        ("未知", "未知", ValidationReason.NOT_IN_VOCAB),
        ("blocked", "blocked", ValidationReason.BLOCKED),
        ("!!水", "!!水", ValidationReason.INVALID_CHARS),
        ("  山 ", "山", None),
        ("ＡＢＣ", "ABC", None),
    ],
)
def test_japanese_normalization_cases(
    service: NormalizationService,
    word: str,
    expected_normalized: str,
    expected_reason: ValidationReason | None,
) -> None:
    result = service.normalize_and_validate(word, "ja")

    assert result.normalized == expected_normalized
    assert result.reason == expected_reason
    assert result.ok is (expected_reason is None)


def test_public_normalize_and_validate_uses_configured_startup_service(
    service: NormalizationService,
) -> None:
    configure_normalizer(service)

    result = normalize_and_validate("Krankenhaus", "de")

    assert result.ok
    assert result.display == "Krankenhaus"
    assert result.lemma == "krankenhaus"


def test_arabic_morphology_provider_resolves_a_root_lemma(
    service: NormalizationService,
) -> None:
    vocabularies = {
        language: index.terms
        for language, index in service.indexes.items()
    }
    vocabularies["ar"] = {"كتب"}
    normalizers = dict(service.normalizers)
    normalizers["ar"] = ArabicNormalizer(
        lemma_provider=lambda word: ("كتب",) if word == "كتبت" else ()
    )
    root_service = NormalizationService(
        vocabularies,
        blocklists={language: () for language in vocabularies},
        normalizers=normalizers,
    )

    result = root_service.normalize_and_validate("كتبت", "ar")

    assert result.ok
    assert result.normalized == "كتب"
    assert result.lemma == "كتب"


def test_blocked_arabic_word_does_not_invoke_morphology(
    service: NormalizationService,
) -> None:
    vocabularies = {
        language: index.terms
        for language, index in service.indexes.items()
    }
    normalizers = dict(service.normalizers)
    normalizers["ar"] = ArabicNormalizer(
        lemma_provider=lambda _word: pytest.fail("blocked words must short-circuit")
    )
    blocked_service = NormalizationService(
        vocabularies,
        blocklists=service.blocked_keys,
        normalizers=normalizers,
    )

    result = blocked_service.normalize_and_validate("ممنوع", "ar")

    assert result.reason is ValidationReason.BLOCKED


def test_japanese_reading_aliases_also_respect_blocklists(
    service: NormalizationService,
) -> None:
    service.blocked_keys["ja"].add("水")

    result = service.normalize_and_validate("ミズ", "ja")

    assert result.reason is ValidationReason.BLOCKED


def test_from_repository_loads_vocabularies_and_blocklists(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    blocklist_root = tmp_path / "blocklists"
    for language, words in {
        "en": ["cat"],
        "de": ["haus"],
        "ar": ["سلام"],
        "es": ["casa"],
        "ja": ["水"],
    }.items():
        (data_root / "vocab").mkdir(parents=True, exist_ok=True)
        (blocklist_root).mkdir(parents=True, exist_ok=True)
        (data_root / "vocab" / f"{language}.json").write_text(
            json.dumps(words, ensure_ascii=False),
            encoding="utf-8",
        )
        (blocklist_root / f"{language}.txt").write_text(
            f"# test list\n{words[0]}\n", encoding="utf-8"
        )

    loaded = NormalizationService.from_repository(
        data_root=data_root,
        blocklist_root=blocklist_root,
    )

    for language, word in {
        "en": "cat",
        "de": "haus",
        "ar": "سلام",
        "es": "casa",
        "ja": "水",
    }.items():
        assert loaded.normalize_and_validate(word, language).reason is ValidationReason.BLOCKED
