#!/usr/bin/env python3
"""Verify claim2:csp graph by exhaustive search.

Claim checked: for every pattern sigma of length k <= 6, the full primal
graph G_sigma is almost nice.  That means: after removing any one x-edge and
any one y-edge, there is a vertex whose deletion leaves a nice graph.

Graphs are multigraphs.  Parallel edges are kept as repeated pairs and count
separately in degrees and cuts.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations


MAX_K = 6
MAX_FAILURES = 5


def edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def inverse(sigma: tuple[int, ...]) -> list[int]:
    inv = [0] * (len(sigma) + 1)
    for i, value in enumerate(sigma, start=1):
        assert inv[value] == 0
        inv[value] = i
    return inv


def primal_edges(sigma: tuple[int, ...]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The full primal graph is the x-path plus the y-path."""
    k = len(sigma)
    inv = inverse(sigma)
    x_edges = [edge(i, i + 1) for i in range(1, k)]
    y_edges = [edge(inv[value], inv[value + 1]) for value in range(1, k)]
    return x_edges, y_edges


def remove_one(edges: list[tuple[int, int]], e: tuple[int, int]) -> list[tuple[int, int]]:
    """Remove exactly one copy of e from the multiset."""
    edges = list(edges)
    assert e in edges
    edges.remove(e)
    return edges


def induced(vertices, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    vertices = set(vertices)
    return [(a, b) for a, b in edges if a in vertices and b in vertices]


def degrees(vertices, edges: list[tuple[int, int]]) -> dict[int, int]:
    """Multigraph degree."""
    deg = {v: 0 for v in vertices}
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def peel(vertices, edges: list[tuple[int, int]]) -> tuple[tuple[int, ...], list[tuple[int, int]]]:
    """Peel vertices of multigraph degree at most one."""
    vertices = set(vertices)
    edges = list(edges)
    while True:
        low = {v for v, d in degrees(vertices, edges).items() if d <= 1}
        if not low:
            return tuple(sorted(vertices)), edges
        vertices -= low
        edges = induced(vertices, edges)


def cycle_subgraph(vertices: tuple[int, ...], edges: list[tuple[int, int]]) -> bool:
    """Underlying simple graph is a subgraph of a c-cycle, c <= 4."""
    if not (2 <= len(vertices) <= 4):
        return False
    return all(d <= 2 for d in degrees(vertices, set(edges)).values())


def underlying_forest(vertices: tuple[int, ...], edges: list[tuple[int, int]]) -> bool:
    """A forest with edge multiplicities: the underlying simple graph is a forest."""
    parent = {v: v for v in vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in set(edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
    return True


def nice(vertices: tuple[int, ...], edges: list[tuple[int, int]]) -> bool:
    """Brute-force the nice partition condition on the peeled graph."""
    if underlying_forest(vertices, edges):
        return True

    for size in range(2, min(4, len(vertices)) + 1):
        for v1_tuple in combinations(vertices, size):
            v1 = set(v1_tuple)
            v2 = set(vertices) - v1
            cut = sum(1 for a, b in edges if (a in v1) != (b in v1))
            if len(v2) < 3 and cut <= 1 and cycle_subgraph(tuple(sorted(v1)), induced(v1, edges)):
                return True
    return False


def separator_graph(
    sigma: tuple[int, ...], hx: int, hy: int
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """Remove the chosen x-edge and y-edge from the full primal graph."""
    x_edges, y_edges = primal_edges(sigma)
    removed_x = edge(hx, hx + 1)
    removed_y = y_edges[hy - 1]
    edges = remove_one(x_edges + y_edges, removed_x)
    edges = remove_one(edges, removed_y)
    return edges, removed_x, removed_y


def deletion_witness(sigma: tuple[int, ...], edges: list[tuple[int, int]]):
    """Try all possible vertices in the 'almost' part of almost nice."""
    vertices = tuple(range(1, len(sigma) + 1))
    for v in vertices:
        remaining = tuple(u for u in vertices if u != v)
        core_vertices, core_edges = peel(remaining, induced(remaining, edges))
        if nice(core_vertices, core_edges):
            return v, core_vertices, core_edges
    return None


def check_case(sigma: tuple[int, ...], hx: int, hy: int):
    edges, removed_x, removed_y = separator_graph(sigma, hx, hy)
    witness = deletion_witness(sigma, edges)
    if witness is not None:
        return None
    return sigma, hx, hy, removed_x, removed_y, edges


def fmt_pattern(sigma: tuple[int, ...]) -> str:
    return "".join(map(str, sigma))


def fmt_edges(edges: list[tuple[int, int]]) -> str:
    counts = Counter(edges)
    return "[" + ", ".join(
        f"{e}" if count == 1 else f"{e}x{count}"
        for e, count in sorted(counts.items())
    ) + "]"


def main() -> int:
    failures = []
    total = 0

    for k in range(2, MAX_K + 1):
        checked = 0
        bad = 0
        for sigma in permutations(range(1, k + 1)):
            for hx in range(1, k):
                for hy in range(1, k):
                    checked += 1
                    failure = check_case(sigma, hx, hy)
                    if failure is None:
                        continue
                    bad += 1
                    if len(failures) < MAX_FAILURES:
                        failures.append(failure)

        total += checked
        print(f"k={k}: checked {checked}, violations {bad}")

    print(f"total checked: {total}")
    if not failures:
        print("VERDICT: PASS")
        return 0

    print(f"VERDICT: FAIL; first {len(failures)} violation(s):")
    for sigma, hx, hy, removed_x, removed_y, edges in failures:
        print(
            f"  sigma={fmt_pattern(sigma)} hx={hx} hy={hy} "
            f"removed={removed_x},{removed_y} edges={fmt_edges(edges)}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
