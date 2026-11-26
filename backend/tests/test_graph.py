"""Tests for pure graph rules and score calculation."""

import pytest

from app.graph import enforce_top_five, final_score, shortest_path


def test_new_stronger_link_prunes_the_weakest_of_five_edges() -> None:
    words = ["hub", "a", "b", "c", "d", "e", "new"]
    prior_weights = {
        ("hub", "a"): 0.9,
        ("hub", "b"): 0.8,
        ("hub", "c"): 0.7,
        ("hub", "d"): 0.6,
        ("hub", "e"): 0.5,
    }

    before = enforce_top_five(words[:-1], prior_weights, 0.4)
    after = enforce_top_five(
        words,
        {**prior_weights, ("hub", "new"): 0.55},
        0.4,
    )

    assert len(before.adjacency["hub"]) == 5
    assert len(after.adjacency["hub"]) == 5
    assert {edge.key for edge in after.pruned_edges} == {("e", "hub")}
    assert ("hub", "new") in {edge.key for edge in after.edges}


def test_top_five_cap_applies_to_both_ends_of_an_edge() -> None:
    words = [
        "hub",
        "leaf",
        *(f"hub{index}" for index in range(5)),
        *(f"leaf{index}" for index in range(5)),
    ]
    weights = {
        ("hub", f"hub{index}"): 0.99 - index * 0.01
        for index in range(5)
    }
    weights.update(
        {
            ("leaf", f"leaf{index}"): 0.98 - index * 0.01
            for index in range(5)
        }
    )
    weights[("leaf", "hub")] = 0.9

    graph = enforce_top_five(words, weights, 0.5)

    assert all(len(neighbors) <= 5 for neighbors in graph.adjacency.values())
    assert ("hub", "leaf") not in {edge.key for edge in graph.edges}


def test_shortest_path_is_deterministic_and_returns_none_when_disconnected() -> None:
    graph = {
        "start": {"b", "a"},
        "a": {"start", "finish"},
        "b": {"start", "finish"},
        "finish": {"a", "b"},
    }

    assert shortest_path(graph, "start", "finish") == ["start", "a", "finish"]
    assert shortest_path({"start": set(), "finish": set()}, "start", "finish") is None


def test_final_score_formula_and_input_validation() -> None:
    assert final_score(4, 0.8, 10) == 44.0
    assert final_score(1, 1.5, 1) == 100.0
    with pytest.raises(ValueError, match="must be positive"):
        final_score(0, 0.5, 1)
