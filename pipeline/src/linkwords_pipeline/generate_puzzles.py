"""Generate daily and practice puzzles from the offline semantic graph."""

from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .config import LANGUAGE_CODES, REPOSITORY_ROOT, get_language_settings
from .lexical import DATABASE_PATH, load_wordnet_reader
from .normalization import normalize_word
from .scoring import score
from .vectors import VectorStore

VOCABULARY_ROOT = REPOSITORY_ROOT / "data" / "vocab"
PUZZLE_ROOT = REPOSITORY_ROOT / "data" / "puzzles"
BLOCKLIST_ROOT = REPOSITORY_ROOT / "shared" / "blocklists"
VALID_PARTS_OF_SPEECH = frozenset({"n", "v", "a", "s"})
MIN_ZIPF_FREQUENCY = 3.0
MAX_ZIPF_FREQUENCY = 6.5
MAX_PATH_STEPS = 6
MIN_PATH_STEPS = 4
REQUIRED_BRIDGE_PATHS = 3
NEAR_SYNONYM_THRESHOLD = 0.80
PairScorer = Callable[[str, str, str], float]
VectorMatrix = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class BridgePaths:
    """Verified shortest and alternative anchor-to-anchor paths."""

    shortest_path: list[str]
    paths: list[list[str]]


@dataclass(frozen=True, slots=True)
class PuzzleGenerationResult:
    """Generated puzzle collections and their candidate rejection report."""

    daily_queue: list[dict[str, Any]]
    practice_pool: list[dict[str, Any]]
    rejected_candidates: list[dict[str, str]]
    rejection_counts: dict[str, int]


def utc_today() -> date:
    """Return the current UTC calendar date for daily puzzle scheduling."""
    return datetime.now(timezone.utc).date()


def load_blocklist(
    language: str,
    *,
    blocklist_root: Path = BLOCKLIST_ROOT,
) -> set[str]:
    """Load and normalize one language's shared blocklist."""
    path = blocklist_root / f"{language}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing language blocklist: {path}")
    return {
        normalize_word(line.strip(), language)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_wordnet_pos_tags(
    language: str,
    vocabulary: set[str],
) -> dict[str, set[str]]:
    """Tag vocabulary lemmas as nouns, verbs, and adjectives from OMW synsets."""
    wordnet, wordnet_languages = load_wordnet_reader(language)

    tags: dict[str, set[str]] = defaultdict(set)
    for synset in tqdm(
        wordnet.all_synsets(),
        desc=f"Tagging anchor vocabulary ({language})",
        disable=not sys.stderr.isatty(),
    ):
        part_of_speech = synset.pos()
        if part_of_speech not in VALID_PARTS_OF_SPEECH:
            continue
        for wordnet_language in wordnet_languages:
            for lemma in synset.lemma_names(lang=wordnet_language):
                word = normalize_word(lemma.replace("_", " "), language)
                if word in vocabulary:
                    tags[word].add(part_of_speech)
    return dict(tags)


def select_anchor_candidates(
    vocabulary: Iterable[str],
    pos_tags: Mapping[str, set[str]],
    frequencies: Mapping[str, float],
    blocked_words: set[str],
    *,
    minimum_frequency: float = MIN_ZIPF_FREQUENCY,
    maximum_frequency: float = MAX_ZIPF_FREQUENCY,
) -> list[str]:
    """Select unblocked vocabulary terms with useful POS and frequency."""
    if minimum_frequency > maximum_frequency:
        raise ValueError("minimum_frequency cannot exceed maximum_frequency")
    return sorted(
        word
        for word in set(vocabulary)
        if word not in blocked_words
        and bool(pos_tags.get(word, set()) & VALID_PARTS_OF_SPEECH)
        and minimum_frequency <= frequencies.get(word, 0.0) <= maximum_frequency
    )


def load_anchor_candidates(
    language: str,
    vocabulary: Sequence[str],
    *,
    blocklist_root: Path = BLOCKLIST_ROOT,
) -> list[str]:
    """Load OMW POS tags and wordfreq frequencies for eligible anchors."""
    from wordfreq import zipf_frequency

    vocabulary_set = set(vocabulary)
    pos_tags = load_wordnet_pos_tags(language, vocabulary_set)
    blocked_words = load_blocklist(language, blocklist_root=blocklist_root)
    frequencies = {
        word: zipf_frequency(word, language)
        for word in vocabulary_set
    }
    return select_anchor_candidates(
        vocabulary_set,
        pos_tags,
        frequencies,
        blocked_words,
    )


def load_lexical_boosts(
    language: str,
    vocabulary: set[str],
    *,
    database_path: Path = DATABASE_PATH,
) -> dict[tuple[str, str], float]:
    """Load all direct lexical pairs meeting the language's edge threshold."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Lexical database not found: {database_path}")
    threshold = get_language_settings(language)["threshold"]
    if threshold is None:
        raise ValueError(f"Connection threshold for {language} has not been calibrated")
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT word_a, word_b, MAX(boost_score)
            FROM synonyms
            WHERE lang = ?
            GROUP BY word_a, word_b
            HAVING MAX(boost_score) >= ?
            """,
            (language, threshold),
        )
        return {
            (word_a, word_b): float(boost)
            for word_a, word_b, boost in rows
            if word_a in vocabulary and word_b in vocabulary
        }


def build_similarity_graph(
    words: Sequence[str],
    vectors: VectorMatrix,
    threshold: float,
    *,
    lexical_boosts: Mapping[tuple[str, str], float] | None = None,
    max_edges: int = 5,
    top_k: int = 5,
    batch_size: int = 128,
) -> dict[str, set[str]]:
    """Build a threshold graph, capped to at most five edges per word."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(words):
        raise ValueError("Vector rows must correspond one-to-one with words")
    if max_edges < 1 or top_k < 1 or batch_size < 1:
        raise ValueError("max_edges, top_k, and batch_size must be positive")
    if len(set(words)) != len(words):
        raise ValueError("Words must be unique")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0) or not np.isfinite(matrix).all():
        raise ValueError("Vectors must be finite and nonzero")
    matrix = matrix / norms[:, None]

    graph: dict[str, set[str]] = {word: set() for word in words}
    if len(words) < 2:
        return graph

    candidate_edges: dict[tuple[int, int], float] = {}
    neighbor_count = min(top_k, len(words) - 1)
    for start in tqdm(
        range(0, len(words), batch_size),
        desc="Finding semantic neighbors",
        disable=not sys.stderr.isatty(),
    ):
        stop = min(start + batch_size, len(words))
        similarities = matrix[start:stop] @ matrix.T
        for local_row, values in enumerate(similarities):
            row = start + local_row
            values[row] = -np.inf
            neighbors = np.argpartition(values, -neighbor_count)[-neighbor_count:]
            neighbors = sorted(
                (int(index) for index in neighbors if values[index] >= threshold),
                key=lambda index: (-float(values[index]), index),
            )
            for neighbor in neighbors:
                edge = (min(row, neighbor), max(row, neighbor))
                candidate_edges[edge] = max(
                    candidate_edges.get(edge, float("-inf")),
                    float(values[neighbor]),
                )

    word_to_index = {word: index for index, word in enumerate(words)}
    for (word_a, word_b), boost in (lexical_boosts or {}).items():
        if word_a in word_to_index and word_b in word_to_index and boost >= threshold:
            first = word_to_index[word_a]
            second = word_to_index[word_b]
            edge = (min(first, second), max(first, second))
            candidate_edges[edge] = max(candidate_edges.get(edge, float("-inf")), boost)

    degree = [0] * len(words)
    sorted_edges = sorted(
        (
            (similarity, first, second)
            for (first, second), similarity in candidate_edges.items()
            if similarity >= threshold
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for _, first, second in sorted_edges:
        if degree[first] >= max_edges or degree[second] >= max_edges:
            continue
        graph[words[first]].add(words[second])
        graph[words[second]].add(words[first])
        degree[first] += 1
        degree[second] += 1
    return graph


def bfs_shortest_path(
    graph: Mapping[str, set[str]],
    start: str,
    goal: str,
) -> list[str] | None:
    """Return one shortest unweighted path using breadth-first search."""
    if start not in graph or goal not in graph:
        return None
    queue = deque([start])
    previous: dict[str, str | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path: list[str] = []
            cursor: str | None = goal
            while cursor is not None:
                path.append(cursor)
                cursor = previous[cursor]
            return list(reversed(path))
        for neighbor in sorted(graph.get(current, set())):
            if neighbor not in previous:
                previous[neighbor] = current
                queue.append(neighbor)
    return None


def _reachable_anchor_targets(
    graph: Mapping[str, set[str]],
    start: str,
    available_anchors: set[str],
) -> list[str]:
    """Find available anchors whose shortest paths are within the puzzle range."""
    queue = deque([(start, 0)])
    distances = {start}
    targets: list[str] = []
    while queue:
        current, distance = queue.popleft()
        if MIN_PATH_STEPS <= distance <= MAX_PATH_STEPS and current in available_anchors:
            targets.append(current)
        if distance >= MAX_PATH_STEPS:
            continue
        for neighbor in sorted(graph.get(current, set())):
            if neighbor not in distances:
                distances.add(neighbor)
                queue.append((neighbor, distance + 1))
    return targets


def _enumerate_paths(
    graph: Mapping[str, set[str]],
    start: str,
    goal: str,
    *,
    maximum_steps: int,
    limit: int,
) -> list[list[str]]:
    queue = deque([(start,)])
    found: list[list[str]] = []
    seen_paths: set[tuple[str, ...]] = set()
    while queue and len(found) < limit:
        path = queue.popleft()
        steps = len(path) - 1
        if steps >= maximum_steps:
            continue
        for neighbor in sorted(graph.get(path[-1], set())):
            if neighbor in path:
                continue
            extended = (*path, neighbor)
            if neighbor == goal:
                if extended not in seen_paths:
                    found.append(list(extended))
                    seen_paths.add(extended)
                    if len(found) >= limit:
                        break
            else:
                queue.append(extended)
    return found


def verify_bridge_paths(
    graph: Mapping[str, set[str]],
    anchor_a: str,
    anchor_b: str,
    *,
    minimum_steps: int = MIN_PATH_STEPS,
    maximum_steps: int = MAX_PATH_STEPS,
    required_paths: int = REQUIRED_BRIDGE_PATHS,
) -> tuple[BridgePaths | None, str | None]:
    """Require a 4-6 edge shortest path and three distinct simple bridge paths."""
    shortest_path = bfs_shortest_path(graph, anchor_a, anchor_b)
    if shortest_path is None:
        return None, "no_path"
    shortest_steps = len(shortest_path) - 1
    if not minimum_steps <= shortest_steps <= maximum_steps:
        return None, "shortest_path_out_of_range"

    paths = _enumerate_paths(
        graph,
        anchor_a,
        anchor_b,
        maximum_steps=maximum_steps,
        limit=required_paths * 20,
    )
    distinct: list[list[str]] = [shortest_path]
    known = {tuple(shortest_path)}
    for path in paths:
        identity = tuple(path)
        if identity not in known:
            distinct.append(path)
            known.add(identity)
        if len(distinct) >= required_paths:
            break
    if len(distinct) < required_paths:
        return None, "fewer_than_three_distinct_paths"
    return BridgePaths(shortest_path=shortest_path, paths=distinct), None


def _difficulty(shortest_steps: int) -> str:
    return {4: "easy", 5: "medium", 6: "hard"}[shortest_steps]


def _puzzle_record(
    language: str,
    anchor_a: str,
    anchor_b: str,
    bridge_paths: BridgePaths,
    puzzle_id: str,
    *,
    puzzle_date: date | None = None,
) -> dict[str, Any]:
    """Convert verified path data to the persisted puzzle schema."""
    steps = len(bridge_paths.shortest_path) - 1
    record: dict[str, Any] = {
        "id": puzzle_id,
        "lang": language,
        "anchorA": anchor_a,
        "anchorB": anchor_b,
        "shortestPath": bridge_paths.shortest_path,
        "alternativePaths": [
            path for path in bridge_paths.paths
            if path != bridge_paths.shortest_path
        ],
        "difficulty": _difficulty(steps),
    }
    if puzzle_date is not None:
        record["date"] = puzzle_date.isoformat()
    return record


def generate_puzzle_collections(
    language: str,
    candidates: Sequence[str],
    graph: Mapping[str, set[str]],
    *,
    daily_count: int = 30,
    practice_count: int = 500,
    seed: int = 42,
    start_date: date | None = None,
    scorer: PairScorer = score,
    connection_threshold: float | None = None,
    near_synonym_threshold: float = NEAR_SYNONYM_THRESHOLD,
    max_attempts_per_puzzle: int = 200,
) -> PuzzleGenerationResult:
    """Choose disjoint anchors and report every rejected candidate pair."""
    if language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language: {language}")
    if daily_count < 0 or practice_count < 0:
        raise ValueError("Puzzle counts cannot be negative")
    target_count = daily_count + practice_count
    if target_count == 0:
        return PuzzleGenerationResult([], [], [], {})

    available = list(dict.fromkeys(word for word in candidates if word in graph))
    if len(available) < target_count * 2:
        raise ValueError(
            f"Need at least {target_count * 2} eligible distinct anchor words; "
            f"found {len(available)}"
        )

    rng = random.Random(seed)
    rng.shuffle(available)
    available_set = set(available)
    available_positions = {word: index for index, word in enumerate(available)}
    attempted_pairs: set[tuple[str, str]] = set()
    exhausted_sources: set[str] = set()
    rejections: list[dict[str, str]] = []
    rejection_counts: Counter[str] = Counter()
    accepted: list[tuple[str, str, BridgePaths]] = []
    attempts_since_accept = 0

    def remove_anchor(word: str) -> None:
        index = available_positions.pop(word)
        last_word = available.pop()
        if index < len(available):
            available[index] = last_word
            available_positions[last_word] = index

    while len(accepted) < target_count and len(available) >= 2:
        if attempts_since_accept >= max_attempts_per_puzzle:
            break
        source_candidates = [
            word for word in available if word not in exhausted_sources
        ]
        if not source_candidates:
            break
        anchor_a = rng.choice(source_candidates)
        target_candidates = _reachable_anchor_targets(
            graph,
            anchor_a,
            available_set,
        )
        if not target_candidates:
            exhausted_sources.add(anchor_a)
            if len(available) == 2:
                anchor_b = next(word for word in available if word != anchor_a)
                pair_key = tuple(sorted((anchor_a, anchor_b)))
                if pair_key not in attempted_pairs:
                    attempted_pairs.add(pair_key)
                    attempts_since_accept += 1
                    _, path_reason = verify_bridge_paths(graph, anchor_a, anchor_b)
                    reason = path_reason or "no_anchor_at_required_distance"
                    rejections.append(
                        {"anchorA": anchor_a, "anchorB": anchor_b, "reason": reason}
                    )
                    rejection_counts[reason] += 1
            else:
                rejection_counts["no_anchor_at_required_distance"] += 1
            continue
        target_candidates = [
            word
            for word in target_candidates
            if tuple(sorted((anchor_a, word))) not in attempted_pairs
        ]
        if not target_candidates:
            exhausted_sources.add(anchor_a)
            continue
        anchor_b = rng.choice(target_candidates)
        pair_key = tuple(sorted((anchor_a, anchor_b)))
        attempted_pairs.add(pair_key)
        attempts_since_accept += 1

        anchor_score = scorer(language, anchor_a, anchor_b)
        if (
            connection_threshold is not None
            and anchor_score >= connection_threshold
        ):
            reason = "anchors_already_connected"
        elif anchor_score >= near_synonym_threshold:
            reason = "near_synonym_anchors"
        else:
            verified, path_reason = verify_bridge_paths(graph, anchor_a, anchor_b)
            if verified is None:
                reason = path_reason or "invalid_graph_paths"
            else:
                accepted.append((anchor_a, anchor_b, verified))
                remove_anchor(anchor_a)
                remove_anchor(anchor_b)
                available_set.discard(anchor_a)
                available_set.discard(anchor_b)
                attempted_pairs.clear()
                exhausted_sources.clear()
                attempts_since_accept = 0
                continue
        rejections.append(
            {"anchorA": anchor_a, "anchorB": anchor_b, "reason": reason}
        )
        rejection_counts[reason] += 1

    if len(accepted) < target_count:
        rejections.append(
            {
                "anchorA": "",
                "anchorB": "",
                "reason": "insufficient_accepted_puzzles",
            }
        )
        rejection_counts["insufficient_accepted_puzzles"] += 1

    effective_start = start_date or utc_today()
    daily_queue = [
        _puzzle_record(
            language,
            anchor_a,
            anchor_b,
            paths,
            f"{language}-daily-{(effective_start + timedelta(days=index)).isoformat()}",
            puzzle_date=effective_start + timedelta(days=index),
        )
        for index, (anchor_a, anchor_b, paths) in enumerate(accepted[:daily_count])
    ]
    practice_pool = [
        _puzzle_record(
            language,
            anchor_a,
            anchor_b,
            paths,
            f"{language}-practice-{index + 1:04d}",
        )
        for index, (anchor_a, anchor_b, paths) in enumerate(
            accepted[daily_count : daily_count + practice_count]
        )
    ]
    return PuzzleGenerationResult(
        daily_queue=daily_queue,
        practice_pool=practice_pool,
        rejected_candidates=rejections,
        rejection_counts=dict(sorted(rejection_counts.items())),
    )


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def generate_language_puzzles(
    language: str,
    *,
    daily_count: int = 30,
    practice_count: int = 500,
    seed: int = 42,
    start_date: date | None = None,
    vocabulary_path: Path | None = None,
    vector_root: Path | None = None,
    database_path: Path = DATABASE_PATH,
    blocklist_root: Path = BLOCKLIST_ROOT,
    output_root: Path = PUZZLE_ROOT,
    scorer: PairScorer = score,
) -> PuzzleGenerationResult:
    """Load local assets, build the semantic graph, and save one language's pools."""
    if language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language: {language}")
    settings = get_language_settings(language)
    threshold = settings["threshold"]
    if threshold is None:
        raise ValueError(
            f"Connection threshold for {language} is unset; run "
            "`python -m pipeline calibrate` first"
        )
    vocabulary_file = vocabulary_path or VOCABULARY_ROOT / f"{language}.json"
    vocabulary: list[str] = json.loads(vocabulary_file.read_text(encoding="utf-8"))
    candidates = load_anchor_candidates(
        language,
        vocabulary,
        blocklist_root=blocklist_root,
    )
    store = VectorStore(
        (vector_root or REPOSITORY_ROOT / "data" / "vectors") / f"{language}.f16.bin",
        (vector_root or REPOSITORY_ROOT / "data" / "vectors")
        / f"{language}.index.json",
    )
    available_words = [word for word in vocabulary if word in store.words]
    row_indices = [store.words[word] for word in available_words]
    matrix = np.asarray(store.matrix[row_indices], dtype=np.float32)
    boosts = load_lexical_boosts(
        language,
        set(available_words),
        database_path=database_path,
    )
    graph = build_similarity_graph(
        available_words,
        matrix,
        float(threshold),
        lexical_boosts=boosts,
    )
    generated = generate_puzzle_collections(
        language,
        candidates,
        graph,
        daily_count=daily_count,
        practice_count=practice_count,
        seed=seed,
        start_date=start_date,
        scorer=scorer,
        connection_threshold=float(threshold),
    )

    payload = {
        "lang": language,
        "daily_queue": generated.daily_queue,
        "practice_pool": generated.practice_pool,
    }
    report = {
        "lang": language,
        "requested": {
            "daily": daily_count,
            "practice": practice_count,
        },
        "generated": {
            "daily": len(generated.daily_queue),
            "practice": len(generated.practice_pool),
        },
        "rejection_counts": generated.rejection_counts,
        "rejected_candidates": generated.rejected_candidates,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_root / f"{language}.json", payload)
    report_path = output_root / f"{language}.report.json"
    _atomic_write_json(report_path, report)
    print(
        f"{language}: generated {len(generated.daily_queue)} daily and "
        f"{len(generated.practice_pool)} practice puzzles"
    )
    print(f"{language}: rejected candidates by reason: {dict(generated.rejection_counts)}")
    if (
        len(generated.daily_queue) != daily_count
        or len(generated.practice_pool) != practice_count
    ):
        raise RuntimeError(
            f"Could not meet requested puzzle counts for {language}; "
            f"see rejection report {report_path}"
        )
    return generated


def generate_languages(
    languages: Sequence[str],
    *,
    daily_count: int = 30,
    practice_count: int = 500,
    seed: int = 42,
    start_date: date | None = None,
    output_root: Path = PUZZLE_ROOT,
) -> dict[str, PuzzleGenerationResult]:
    """Generate requested language puzzle sets with per-language deterministic seeds."""
    effective_start = start_date or utc_today()
    return {
        language: generate_language_puzzles(
            language,
            daily_count=daily_count,
            practice_count=practice_count,
            seed=seed + LANGUAGE_CODES.index(language),
            start_date=effective_start,
            output_root=output_root,
        )
        for language in languages
    }
