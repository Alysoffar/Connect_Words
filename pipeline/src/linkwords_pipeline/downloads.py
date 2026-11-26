"""Explicit acquisition of pinned language data and model files."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from tqdm import tqdm

from .config import LANGUAGE_CODES, REPOSITORY_ROOT
from .vectors import BGE_MODEL_ID, BGE_MODEL_REVISION, MODEL_ROOT, NUMBERBATCH_URL

SOURCES_ROOT = REPOSITORY_ROOT / "data" / "sources"
FILTERS_ROOT = REPOSITORY_ROOT / "data" / "filters"
NLTK_DATA_ROOT = REPOSITORY_ROOT / "data" / "nltk_data"
STOPWORDS_URL = (
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-iso/"
    "6e8e9a2d1c019ad2a885bcbe157628dea624dfb6/"
    "python/stopwordsiso/stopwords-iso.json"
)
PROFANITY_REVISION = "5faf2ba42d7b1c0977169ec3611df25a3c08eb13"
PROFANITY_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
    f"{PROFANITY_REVISION}/{{language}}"
)
LEMMA_LIST_URLS = {
    "de": (
        "https://raw.githubusercontent.com/wooorm/dictionaries/"
        "8cfea406b505e4d7df52d5a19bce525df98c54ab/dictionaries/de/index.dic"
    ),
    "es": (
        "https://raw.githubusercontent.com/wooorm/dictionaries/"
        "8cfea406b505e4d7df52d5a19bce525df98c54ab/dictionaries/es/index.dic"
    ),
    "ar": (
        "https://raw.githubusercontent.com/linuxscout/ayaspell/"
        "36a90abb22c5033be58e709c8980160e0ce6c88c/dict/builddict/arb.dic"
    ),
}


def download_file(url: str, destination: Path) -> None:
    """Download a URL atomically, resuming a partial file when supported."""
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".part")
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request) as response:
        resumed = offset > 0 and response.status == 206
        if not resumed:
            offset = 0
        content_length = response.headers.get("Content-Length")
        total = offset + int(content_length) if content_length else None
        mode = "ab" if resumed else "wb"
        with partial_path.open(mode) as output, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                progress.update(len(block))
    os.replace(partial_path, destination)


def download_data(languages: tuple[str, ...]) -> None:
    """Fetch required sources, filters, NLTK corpora, and pinned BGE-M3 files."""
    if not languages or any(language not in LANGUAGE_CODES for language in languages):
        raise ValueError(f"Languages must be selected from {', '.join(LANGUAGE_CODES)}")

    download_file(
        NUMBERBATCH_URL,
        SOURCES_ROOT / "numberbatch-19.08.txt.gz",
    )
    stopword_path = FILTERS_ROOT / "stopwords-iso.json"
    download_file(STOPWORDS_URL, stopword_path)
    stopword_data = json.loads(stopword_path.read_text(encoding="utf-8"))
    for language in languages:
        if language not in stopword_data:
            raise ValueError(f"Stopwords ISO has no list for language {language}")
        download_file(
            PROFANITY_URL.format(language=language),
            FILTERS_ROOT / "profanity" / f"{language}.txt",
        )
        lemma_list_url = LEMMA_LIST_URLS.get(language)
        if lemma_list_url is not None:
            download_file(
                lemma_list_url,
                SOURCES_ROOT / "lemmas" / f"{language}.dic",
            )

    import nltk

    NLTK_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DATA_ROOT))
    for resource in ("wordnet", "omw-1.4", "extended_omw"):
        if not nltk.download(resource, download_dir=str(NLTK_DATA_ROOT), raise_on_error=True):
            raise RuntimeError(f"Could not download NLTK resource: {resource}")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=BGE_MODEL_ID,
        revision=BGE_MODEL_REVISION,
        local_dir=str(MODEL_ROOT),
        ignore_patterns=["onnx/**"],
    )


def load_stopwords(language: str) -> set[str]:
    """Load a downloaded Stopwords ISO list for one language."""
    path = FILTERS_ROOT / "stopwords-iso.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run `python -m pipeline download --lang {language}`"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return set(data[language])
    except KeyError as error:
        raise ValueError(f"Stopwords ISO has no list for language {language}") from error


def load_profanities(language: str) -> set[str]:
    """Load a downloaded profanity list for one language."""
    path = FILTERS_ROOT / "profanity" / f"{language}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run `python -m pipeline download --lang {language}`"
        )
    return {
        word.strip()
        for word in path.read_text(encoding="utf-8").splitlines()
        if word.strip()
    }


def load_supplemental_lemmas(language: str) -> set[str]:
    """Load headwords from the configured language dictionary, if available."""
    if language not in LEMMA_LIST_URLS:
        return set()
    path = SOURCES_ROOT / "lemmas" / f"{language}.dic"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run `python -m pipeline download --lang {language}`"
        )

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"Lemma list is empty: {path}")
    return {
        entry.split("/", maxsplit=1)[0].strip()
        for entry in lines[1:]
        if entry.strip() and not entry.lstrip().startswith("#")
    }


def require_vector_sources() -> tuple[Path, Path]:
    """Return vector source paths or explain which explicit download is needed."""
    numberbatch_path = SOURCES_ROOT / "numberbatch-19.08.txt.gz"
    if not numberbatch_path.is_file():
        raise FileNotFoundError(
            f"Missing {numberbatch_path}; run `python -m pipeline download --lang all`"
        )
    if not (MODEL_ROOT / "modules.json").is_file():
        raise FileNotFoundError(
            f"Missing BGE-M3 model at {MODEL_ROOT}; "
            "run `python -m pipeline download --lang all`"
        )
    return numberbatch_path, MODEL_ROOT
