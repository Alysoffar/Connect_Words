"""Pure graph rules for LinkWords board construction and path solving."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


MAX_EDGES_PER_NODE = 5


@dataclass(frozen=True, slots=True)
class Edge:
    """Undirected weighted edge between two normalized board words."""

    word_a: str
    word_b: str
    weight: float

    @property
    def key(self) -> tuple[str, str]:
        """Return the canonical undirected edge key."""
        return canonical_edge_key(self.word_a, self.word_b)


@dataclass(frozen=True, slots=True)
class GraphState:
    """Selected edges and eligible edges removed by the degree cap."""

    edges: tuple[Edge, ...]
    pruned_edges: tuple[Edge, ...]

    @property
    def adjacency(self) -> dict[str, set[str]]:
        """Return the selected graph as an adjacency map."""
        graph: dict[str, set[str]] = {}
        for edge in self.edges:
            graph.setdefault(edge.word_a, set()).add(edge.word_b)
            graph.setdefault(edge.word_b, set()).add(edge.word_a)
        return graph


def canonical_edge_key(word_a: str, word_b: str) -> tuple[str, str]:
    """Return a stable key for an undirected pair."""
    return (word_a, word_b) if word_a <= word_b else (word_b, word_a)


def enforce_top_five(
    words: Sequence[str],
    edge_weights: Mapping[tuple[str, str], float],
    threshold: float,
    *,
    max_edges: int = MAX_EDGES_PER_NODE,
) -> GraphState:
    """Keep the strongest eligible edges without exceeding a node's degree cap."""
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    word_set = set(words)
    if len(word_set) != len(words):
        raise ValueError("words must be unique")

    eligible: dict[tuple[str, str], float] = {}
    for (word_a, word_b), weight in edge_weights.items():
        if word_a == word_b or word_a not in word_set or word_b not in word_set:
            continue
        if not math.isfinite(weight):
            raise ValueError("edge weights must be finite")
        key = canonical_edge_key(word_a, word_b)
        if weight >= threshold:
            eligible[key] = max(weight, eligible.get(key, float("-inf")))

    ordered = sorted(
        eligible.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    degrees = {word: 0 for word in words}
    selected: list[Edge] = []
    selected_keys: set[tuple[str, str]] = set()
    for (word_a, word_b), weight in ordered:
        if degrees[word_a] >= max_edges or degrees[word_b] >= max_edges:
            continue
        selected.append(Edge(word_a, word_b, weight))
        selected_keys.add((word_a, word_b))
        degrees[word_a] += 1
        degrees[word_b] += 1

    pruned = tuple(
        Edge(word_a, word_b, weight)
        for (word_a, word_b), weight in ordered
        if (word_a, word_b) not in selected_keys
    )
    return GraphState(tuple(selected), pruned)


def shortest_path(
    graph: Mapping[str, set[str]],
    start: str,
    goal: str,
) -> list[str] | None:
    """Return one deterministic shortest path, or None when disconnected."""
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


def final_score(
    path_length: int,
    average_edge_weight: float,
    total_guesses: int,
) -> float:
    """Score a solve on a 0-100 scale; see backend README for the formula."""
    if path_length < 1 or total_guesses < 1:
        raise ValueError("path_length and total_guesses must be positive")
    if not math.isfinite(average_edge_weight):
        raise ValueError("average_edge_weight must be finite")
    average_edge_weight = min(1.0, max(0.0, average_edge_weight))
    value = 100.0 * (
        0.4 / path_length
        + 0.4 * average_edge_weight
        + 0.2 / total_guesses
    )
    return round(min(100.0, max(0.0, value)), 2)
