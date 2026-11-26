"""Build filtered, frequency-ranked vocabularies from wordfreq."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

from .config import REPOSITORY_ROOT, get_language_settings
from .normalization import allows_single_character_word, normalize_word

URL_PATTERN = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\.?$")
JOINERS = {"'", "\u2019", "-"}
DEFAULT_VOCABULARY_SIZE = 40_000
MAX_CANDIDATES = 250_000
VOCABULARY_FILTER_VERSION = 4
LOGGER = logging.getLogger(__name__)


def _log_removals(language: str, removals: Mapping[str, int]) -> None:
    """Report the number of rejected candidates for every vocabulary rule."""
    LOGGER.info(
        "%s vocabulary rule removals: %s",
        language,
        ", ".join(f"{rule}={count}" for rule, count in removals.items()),
    )


def _is_word_shape_valid(word: str, language: str) -> bool:
    if not word or URL_PATTERN.match(word) or any(character.isdigit() for character in word):
        return False
    if allows_single_character_word(word, language):
        return True

    has_letter = False
    for index, character in enumerate(word):
        category = unicodedata.category(character)
        if category.startswith("L"):
            has_letter = True
        elif category.startswith("M"):
            continue
        elif character in JOINERS and 0 < index < len(word) - 1:
            continue
        else:
            return False
    return has_letter


def is_clean_word(
    candidate: str,
    language: str,
    *,
    stopwords: set[str] | frozenset[str] = frozenset(),
    profanities: set[str] | frozenset[str] = frozenset(),
    proper_nouns: set[str] | frozenset[str] = frozenset(),
    known_lemmas: set[str] | frozenset[str] | None = None,
) -> bool:
    """Check normalized spelling, configured length, and exclusion lists."""
    settings = get_language_settings(language)
    stripped = candidate.strip()
    if URL_PATTERN.match(stripped) or DOMAIN_PATTERN.fullmatch(stripped):
        return False
    word = normalize_word(candidate, language)
    exclusions = (
        {normalize_word(item, language) for item in stopwords},
        {normalize_word(item, language) for item in profanities},
        {normalize_word(item, language) for item in proper_nouns},
    )
    if not _is_clean_normalized_word(word, language, settings, exclusions):
        return False
    if known_lemmas is not None:
        normalized_lemmas = {
            normalize_word(lemma.replace("_", " "), language)
            for lemma in known_lemmas
        }
        return word in normalized_lemmas
    return True


def _is_clean_normalized_word(
    word: str,
    language: str,
    settings: Mapping[str, Any],
    exclusions: tuple[set[str], set[str], set[str]],
) -> bool:
    """Validate a normalized candidate against already-normalized inputs."""
    if not _is_word_shape_valid(word, language):
        return False
    if len(word) > settings["max_word_length"]:
        return False
    if (
        len(word) == 1
        and settings["min_word_length"] == 1
        and not allows_single_character_word(word, language)
    ):
        return False
    if len(word) < settings["min_word_length"] and not (
        allows_single_character_word(word, language)
    ):
        return False
    return all(word not in excluded for excluded in exclusions)


def load_word_list(path: Path) -> set[str]:
    """Read one UTF-8 word per line, ignoring blank lines and comments."""
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def build_vocabulary(
    language: str,
    *,
    output_path: Path | None = None,
    target_size: int = DEFAULT_VOCABULARY_SIZE,
    candidates: Sequence[str] | None = None,
    stopwords: Iterable[str] = (),
    profanities: Iterable[str] = (),
    proper_nouns: Iterable[str] = (),
    known_lemmas: Iterable[str] | None = None,
) -> list[str]:
    """Build and write a clean, de-duplicated frequency-ordered vocabulary."""
    if not 30_000 <= target_size <= 50_000:
        raise ValueError("target_size must be between 30,000 and 50,000")

    if candidates is None:
        from importlib.metadata import version
        from wordfreq import top_n_list

        candidates = top_n_list(language, MAX_CANDIDATES)
        frequency_source = f"wordfreq-{version('wordfreq')}"
    else:
        frequency_source = hashlib.sha256(
            json.dumps(list(candidates), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    if known_lemmas is None:
        from .lexical import load_wordnet_lemmas

        known_lemmas = load_wordnet_lemmas(language)
    lemma_set = {
        normalize_word(lemma.replace("_", " "), language)
        for lemma in known_lemmas
    }
    stopword_set = {normalize_word(word, language) for word in stopwords}
    profanity_set = {normalize_word(word, language) for word in profanities}
    proper_noun_set = {normalize_word(word, language) for word in proper_nouns}
    settings = get_language_settings(language)
    filter_digest = hashlib.sha256(
        json.dumps(
            [
                sorted(stopword_set),
                sorted(profanity_set),
                sorted(proper_noun_set),
                sorted(lemma_set),
            ],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    destination = output_path or REPOSITORY_ROOT / "data" / "vocab" / f"{language}.json"
    manifest_path = destination.with_suffix(".manifest.json")
    manifest = {
        "language": language,
        "target_size": target_size,
        "frequency_source": frequency_source,
        "filter_sha256": filter_digest,
        "filter_version": VOCABULARY_FILTER_VERSION,
        "capitalization_exclusion": settings["exclude_capitalized_candidates"],
    }
    if destination.is_file() and manifest_path.is_file():
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(cached_manifest.get(key) == value for key, value in manifest.items()):
            cached_words = json.loads(destination.read_text(encoding="utf-8"))
            cached_removals = cached_manifest.get("removals")
            if (
                len(cached_words) >= min(30_000, target_size)
                and isinstance(cached_removals, dict)
            ):
                _log_removals(language, cached_removals)
                return cached_words

    words: list[str] = []
    seen: set[str] = set()
    removals = {
        "capitalization": 0,
        "duplicate": 0,
        "word_shape": 0,
        "length": 0,
        "stopword": 0,
        "profanity": 0,
        "proper_noun_list": 0,
        "not_in_lexicon": 0,
    }
    for candidate in tqdm(
        candidates,
        desc=f"Filtering vocabulary ({language})",
        disable=not sys.stderr.isatty(),
    ):
        if settings["exclude_capitalized_candidates"] and candidate[:1].isupper():
            removals["capitalization"] += 1
            continue
        stripped = candidate.strip()
        if URL_PATTERN.match(stripped) or DOMAIN_PATTERN.fullmatch(stripped):
            removals["word_shape"] += 1
            continue
        word = normalize_word(candidate, language)
        if word in seen:
            removals["duplicate"] += 1
            continue
        if not _is_word_shape_valid(word, language):
            removals["word_shape"] += 1
            continue
        if len(word) > settings["max_word_length"] or (
            len(word) < settings["min_word_length"]
            and not allows_single_character_word(word, language)
        ):
            removals["length"] += 1
            continue
        if word in stopword_set:
            removals["stopword"] += 1
            continue
        if word in profanity_set:
            removals["profanity"] += 1
            continue
        if word in proper_noun_set:
            removals["proper_noun_list"] += 1
            continue
        if word not in lemma_set:
            removals["not_in_lexicon"] += 1
            continue
        words.append(word)
        seen.add(word)
        if len(words) == target_size:
            break

    if len(words) < min(30_000, target_size):
        raise ValueError(
            f"Only {len(words)} clean words found for {language}; "
            "at least 30,000 are required for a production vocabulary"
        )

    _log_removals(language, removals)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(words, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({**manifest, "removals": removals}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return words
