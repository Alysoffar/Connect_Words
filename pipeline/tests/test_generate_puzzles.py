"""Tests for puzzle graph constraints and anchor eligibility."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from linkwords_pipeline.generate_puzzles import (
    build_similarity_graph,
    generate_puzzle_collections,
    generate_language_puzzles,
    select_anchor_candidates,
    verify_bridge_paths,
)


def _three_route_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {"start": set(), "finish": set()}
    for route_number in range(3):
        middle = [
            f"r{route_number}-a",
            f"r{route_number}-b",
            f"r{route_number}-c",
        ]
        route = ["start", *middle, "finish"]
        for first, second in zip(route, route[1:]):
            graph.setdefault(first, set()).add(second)
            graph.setdefault(second, set()).add(first)
    return graph


def test_bfs_accepts_three_distinct_four_step_bridge_paths() -> None:
    verified, reason = verify_bridge_paths(
        _three_route_graph(),
        "start",
        "finish",
    )

    assert reason is None
    assert verified is not None
    assert len(verified.shortest_path) - 1 == 4
    assert len(verified.paths) == 3
    assert len({tuple(path) for path in verified.paths}) == 3


def test_bfs_rejects_a_shortest_path_below_four_steps() -> None:
    graph = _three_route_graph()
    graph["start"].add("finish")
    graph["finish"].add("start")

    verified, reason = verify_bridge_paths(graph, "start", "finish")

    assert verified is None
    assert reason == "shortest_path_out_of_range"


def test_bfs_rejects_two_paths_even_when_shortest_length_is_valid() -> None:
    graph = _three_route_graph()
    third_path = ["start", "r2-a", "r2-b", "r2-c", "finish"]
    for first, second in zip(third_path, third_path[1:]):
        graph[first].discard(second)
        graph[second].discard(first)

    verified, reason = verify_bridge_paths(graph, "start", "finish")

    assert verified is None
    assert reason == "fewer_than_three_distinct_paths"


def test_bfs_reports_disconnected_anchors() -> None:
    verified, reason = verify_bridge_paths(
        {"start": set(), "finish": set()},
        "start",
        "finish",
    )

    assert verified is None
    assert reason == "no_path"


def test_similarity_graph_enforces_threshold_and_top_five_degree() -> None:
    vectors = np.eye(8, dtype=np.float32)

    graph = build_similarity_graph(
        [f"word{index}" for index in range(8)],
        vectors,
        threshold=0.1,
        lexical_boosts={
            ("word0", "word1"): 0.95,
            ("word0", "word2"): 0.90,
            ("word0", "word3"): 0.85,
        },
        max_edges=2,
        top_k=2,
    )

    assert max(map(len, graph.values())) <= 2
    assert "word1" in graph["word0"]
    assert len(graph["word0"]) == 2


def test_anchor_selection_requires_pos_frequency_and_not_blocked() -> None:
    candidates = select_anchor_candidates(
        ["animal", "run", "blocked", "rare", "adverb"],
        {
            "animal": {"n"},
            "run": {"v"},
            "blocked": {"n"},
            "rare": {"a"},
            "adverb": {"r"},
        },
        {
            "animal": 4.1,
            "run": 3.7,
            "blocked": 5.0,
            "rare": 2.0,
            "adverb": 4.0,
        },
        {"blocked"},
    )

    assert candidates == ["animal", "run"]


def test_anchor_selection_rejects_inverted_frequency_range() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        select_anchor_candidates([], {}, {}, set(), minimum_frequency=5, maximum_frequency=2)


def test_generated_daily_and_practice_puzzles_use_unique_anchors_and_dates() -> None:
    candidates = [f"anchor{index}" for index in range(6)]
    graph: dict[str, set[str]] = {}
    for pair_index in range(3):
        start, finish = candidates[pair_index * 2 : pair_index * 2 + 2]
        for route_index in range(3):
            route = [
                start,
                f"bridge{pair_index}-{route_index}-a",
                f"bridge{pair_index}-{route_index}-b",
                f"bridge{pair_index}-{route_index}-c",
                finish,
            ]
            for first, second in zip(route, route[1:]):
                graph.setdefault(first, set()).add(second)
                graph.setdefault(second, set()).add(first)

    generated = generate_puzzle_collections(
        "en",
        candidates,
        graph,
        daily_count=1,
        practice_count=2,
        seed=3,
        start_date=date(2026, 11, 26),
        scorer=lambda _language, _a, _b: 0.0,
    )

    puzzles = [*generated.daily_queue, *generated.practice_pool]
    anchors = [
        anchor
        for puzzle in puzzles
        for anchor in (puzzle["anchorA"], puzzle["anchorB"])
    ]
    assert len(puzzles) == 3
    assert len(set(anchors)) == len(anchors)
    assert generated.daily_queue[0]["date"] == "2026-11-26"
    assert all(len(puzzle["alternativePaths"]) >= 2 for puzzle in puzzles)


def test_default_daily_date_uses_utc_today(monkeypatch: pytest.MonkeyPatch) -> None:
    import linkwords_pipeline.generate_puzzles as puzzle_module

    monkeypatch.setattr(puzzle_module, "utc_today", lambda: date(2026, 11, 26))
    graph = _three_route_graph()

    generated = generate_puzzle_collections(
        "en",
        ["start", "finish"],
        graph,
        daily_count=1,
        practice_count=0,
        seed=3,
        scorer=lambda _language, _a, _b: 0.0,
    )

    assert generated.daily_queue[0]["date"] == "2026-11-26"


def test_anchor_search_skips_disconnected_candidates() -> None:
    candidates = ["disconnected-a", "disconnected-b", "start", "finish"]
    graph = _three_route_graph()
    graph["disconnected-a"] = set()
    graph["disconnected-b"] = set()

    generated = generate_puzzle_collections(
        "en",
        candidates,
        graph,
        daily_count=1,
        practice_count=0,
        seed=3,
        scorer=lambda _language, _a, _b: 0.0,
        max_attempts_per_puzzle=1,
    )

    assert len(generated.daily_queue) == 1
    assert generated.daily_queue[0]["anchorA"] in {"start", "finish"}
    assert generated.daily_queue[0]["anchorB"] in {"start", "finish"}


def test_puzzle_generation_rejects_anchors_already_above_connection_threshold() -> None:
    generated = generate_puzzle_collections(
        "en",
        ["start", "finish"],
        _three_route_graph(),
        daily_count=1,
        practice_count=0,
        scorer=lambda _language, _a, _b: 0.6,
        connection_threshold=0.5,
    )

    assert generated.daily_queue == []
    assert generated.rejection_counts["anchors_already_connected"] == 1


def test_incomplete_generation_saves_report_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linkwords_pipeline.generate_puzzles as puzzle_module

    vocabulary_path = tmp_path / "vocab.json"
    vocabulary_path.write_text(json.dumps(["start", "finish"]), encoding="utf-8")

    class FakeVectorStore:
        words = {"start": 0, "finish": 1}
        matrix = np.eye(2, dtype=np.float32)

        def __init__(self, _matrix_path: Path, _index_path: Path) -> None:
            pass

    monkeypatch.setattr(puzzle_module, "get_language_settings", lambda _lang: {"threshold": 0.5})
    monkeypatch.setattr(puzzle_module, "load_anchor_candidates", lambda *_args, **_kwargs: ["start", "finish"])
    monkeypatch.setattr(puzzle_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(puzzle_module, "load_lexical_boosts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        puzzle_module,
        "build_similarity_graph",
        lambda words, _matrix, _threshold, **_kwargs: {word: set() for word in words},
    )

    with pytest.raises(RuntimeError, match="Could not meet requested puzzle counts"):
        generate_language_puzzles(
            "en",
            daily_count=1,
            practice_count=0,
            vocabulary_path=vocabulary_path,
            vector_root=tmp_path / "vectors",
            output_root=tmp_path / "puzzles",
            scorer=lambda *_args: 0.0,
        )

    report = json.loads(
        (tmp_path / "puzzles" / "en.report.json").read_text(encoding="utf-8")
    )
    assert report["generated"]["daily"] == 0
    assert report["rejection_counts"]["no_path"] == 1
