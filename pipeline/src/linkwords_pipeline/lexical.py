"""Extract Open Multilingual WordNet associations into SQLite."""

from __future__ import annotations

import hashlib
import itertools
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from .config import LANGUAGE_CODES, REPOSITORY_ROOT
from .normalization import WORDNET_LANGUAGE_CODES, normalize_word

DATABASE_PATH = REPOSITORY_ROOT / "data" / "linkwords.sqlite"
SYNONYM_SCORE = 0.90
TAXONOMIC_SCORE = 0.75
LEXICAL_DATA_VERSION = "omw-1.4+extended_omw"


class WordNetSynset(Protocol):
    """Small interface used from NLTK's WordNet synset objects."""

    def lemma_names(self, lang: str) -> list[str]: ...

    def hypernyms(self) -> list["WordNetSynset"]: ...

    def hyponyms(self) -> list["WordNetSynset"]: ...


class WordNetReader(Protocol):
    """Multilingual WordNet operations used by pipeline stages."""

    def ensure_loaded(self) -> None: ...

    def add_exomw(self) -> None: ...

    def langs(self) -> list[str]: ...

    def all_lemma_names(
        self,
        pos: str | None = None,
        lang: str = "eng",
    ) -> Iterable[str]: ...

    def all_synsets(self, pos: str | None = None) -> Iterable[WordNetSynset]: ...


def _matching_wordnet_languages(
    language: str,
    available_languages: Iterable[str],
) -> tuple[str, ...]:
    """Select standard and extended OMW codes for a configured language."""
    try:
        base_code = WORDNET_LANGUAGE_CODES[language]
    except KeyError as error:
        raise ValueError(f"Unsupported language: {language}") from error
    return tuple(
        sorted(
            code
            for code in set(available_languages)
            if code == base_code or code.startswith(f"{base_code}_")
        )
    )


def load_wordnet_reader(language: str) -> tuple[WordNetReader, tuple[str, ...]]:
    """Load OMW and extended OMW aliases for one configured language."""
    if language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language: {language}")
    import nltk

    nltk_data_path = str(REPOSITORY_ROOT / "data" / "nltk_data")
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)
    try:
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
        available_languages = set(wordnet.langs())
        if not any(
            code.endswith(("_cldr", "_wikt")) for code in available_languages
        ):
            wordnet.add_exomw()
            available_languages = set(wordnet.langs())
    except LookupError as error:
        raise RuntimeError(
            "WordNet data is missing; run `python -m pipeline download --lang "
            f"{language}` first"
        ) from error

    language_codes = _matching_wordnet_languages(language, available_languages)
    if not language_codes:
        raise RuntimeError(
            f"Open Multilingual WordNet has no '{WORDNET_LANGUAGE_CODES[language]}' "
            f"data. Available codes: {', '.join(sorted(available_languages))}"
        )
    return wordnet, language_codes


def canonical_pair(word_a: str, word_b: str) -> tuple[str, str]:
    """Return a stable undirected ordering for a lexical association pair."""
    return (word_a, word_b) if word_a <= word_b else (word_b, word_a)


def create_lexical_schema(connection: sqlite3.Connection) -> None:
    """Create the association table and indexes if they do not exist."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS synonyms (
            lang TEXT NOT NULL,
            word_a TEXT NOT NULL,
            word_b TEXT NOT NULL,
            relation TEXT NOT NULL,
            boost_score REAL NOT NULL CHECK (boost_score BETWEEN 0.0 AND 1.0),
            PRIMARY KEY (lang, word_a, word_b, relation)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_synonyms_lookup "
        "ON synonyms (lang, word_a, word_b)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS lexical_builds (
            lang TEXT PRIMARY KEY,
            vocabulary_sha256 TEXT NOT NULL,
            data_version TEXT NOT NULL
        )
        """
    )


def lexical_boost_score(
    language: str,
    word_a: str,
    word_b: str,
    database_path: Path = DATABASE_PATH,
) -> float:
    """Return the highest direct lexical relation boost for a pair."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Lexical database not found: {database_path}")
    first, second = canonical_pair(
        normalize_word(word_a, language),
        normalize_word(word_b, language),
    )
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(boost_score), 0.0)
            FROM synonyms
            WHERE lang = ? AND word_a = ? AND word_b = ?
            """,
            (language, first, second),
        ).fetchone()
    return float(row[0])


def _insert_pair(
    connection: sqlite3.Connection,
    language: str,
    word_a: str,
    word_b: str,
    relation: str,
    score: float,
    vocabulary: set[str] | None,
) -> None:
    first = normalize_word(word_a.replace("_", " "), language)
    second = normalize_word(word_b.replace("_", " "), language)
    if not first or not second or first == second:
        return
    if vocabulary is not None and (first not in vocabulary or second not in vocabulary):
        return
    first, second = canonical_pair(first, second)
    connection.execute(
        """
        INSERT INTO synonyms (lang, word_a, word_b, relation, boost_score)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (lang, word_a, word_b, relation)
        DO UPDATE SET boost_score = MAX(boost_score, excluded.boost_score)
        """,
        (language, first, second, relation, score),
    )


def _synset_words(synset: WordNetSynset, wordnet_language: str) -> list[str]:
    return list(synset.lemma_names(lang=wordnet_language))


def load_wordnet_lemmas(language: str) -> set[str]:
    """Load normalized OMW lemmas for one language from NLTK's WordNet reader."""
    wordnet, wordnet_languages = load_wordnet_reader(language)
    return {
        normalize_word(lemma.replace("_", " "), language)
        for wordnet_language in wordnet_languages
        for lemma in wordnet.all_lemma_names(lang=wordnet_language)
    }


def build_wordnet_database(
    language: str,
    *,
    database_path: Path = DATABASE_PATH,
    vocabulary: Iterable[str] | None = None,
    force: bool = False,
) -> int:
    """Build synonyms and direct taxonomy pairs for one language."""
    wordnet, wordnet_languages = load_wordnet_reader(language)

    normalized_vocabulary = (
        {normalize_word(word, language) for word in vocabulary}
        if vocabulary is not None
        else None
    )
    vocabulary_hash = hashlib.sha256(
        "\n".join(sorted(normalized_vocabulary or ())).encode("utf-8")
    ).hexdigest()

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        create_lexical_schema(connection)
        existing = connection.execute(
            "SELECT vocabulary_sha256, data_version FROM lexical_builds WHERE lang = ?",
            (language,),
        ).fetchone()
        if not force and existing == (vocabulary_hash, LEXICAL_DATA_VERSION):
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM synonyms WHERE lang = ?", (language,)
                ).fetchone()[0]
            )

        connection.execute("DELETE FROM synonyms WHERE lang = ?", (language,))
        synsets = list(wordnet.all_synsets())
        for synset in tqdm(
            synsets,
            desc=f"Extracting WordNet ({language})",
            disable=not sys.stderr.isatty(),
        ):
            words = list(
                dict.fromkeys(
                    word
                    for wordnet_language in wordnet_languages
                    for word in _synset_words(synset, wordnet_language)
                )
            )
            if normalized_vocabulary is not None:
                words = [
                    word for word in words
                    if normalize_word(word.replace("_", " "), language) in normalized_vocabulary
                ]
            for word_a, word_b in itertools.combinations(words, 2):
                _insert_pair(
                    connection,
                    language,
                    word_a,
                    word_b,
                    "synonym",
                    SYNONYM_SCORE,
                    normalized_vocabulary,
                )

            for relation, related_synsets in (
                ("hypernym", synset.hypernyms()),
                ("hyponym", synset.hyponyms()),
            ):
                for related_synset in related_synsets:
                    related_words = list(
                        dict.fromkeys(
                            word
                            for wordnet_language in wordnet_languages
                            for word in _synset_words(
                                related_synset,
                                wordnet_language,
                            )
                        )
                    )
                    for word_a in words:
                        for word_b in related_words:
                            _insert_pair(
                                connection,
                                language,
                                word_a,
                                word_b,
                                relation,
                                TAXONOMIC_SCORE,
                                normalized_vocabulary,
                            )

        connection.execute(
            """
            INSERT INTO lexical_builds (lang, vocabulary_sha256, data_version)
            VALUES (?, ?, ?)
            ON CONFLICT (lang) DO UPDATE SET
                vocabulary_sha256 = excluded.vocabulary_sha256,
                data_version = excluded.data_version
            """,
            (language, vocabulary_hash, LEXICAL_DATA_VERSION),
        )
        connection.commit()
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM synonyms WHERE lang = ?", (language,)
            ).fetchone()[0]
        )
