#!/usr/bin/env python3
"""Compute extra Y1 guesses needed to linearize supplied 3-round Y2 conditions."""

from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Sequence, Set, Tuple

LANES = 4
LANE_BITS = 32
PLANES = 3

Y1Var = Tuple[int, int, int]
Monomial = frozenset[Y1Var]
Polynomial = Set[Monomial]
Edge = Tuple[Y1Var, Y1Var]

Y1_COND_0 = {
    0: {
        0: [0, 4, 5, 9, 14, 18, 27, 28],
        1: [0, 5, 9, 14, 18, 23],
        2: [23],
        3: [0, 4, 8, 13, 17, 22, 23, 26, 27, 31],
    },
    1: {0: [1, 6, 10, 15, 19, 24], 2: [8, 31]},
    2: {1: [8, 31], 2: [2, 11], 3: [7, 16, 25]},
}

Y1_COND_1 = {
    0: {
        1: [1, 6, 15, 24],
        2: [2, 11],
        3: [7, 16, 25],
    },
    1: {
        0: [0, 4, 8, 9, 13, 18, 23, 27, 31],
        1: [4, 13],
        3: [4, 13, 18, 22, 27],
    },
    2: {0: [1, 10, 19, 24], 1: [21, 30], 2: [8, 31], 3: [5, 14, 28]},
}

Y2_COND_0 = {
    1: {0: [2, 11, 20, 25, 29]},
}

Y2_COND_1 = {
    0: {0: [3, 17, 26], 2: [2, 11, 25], 3: [12, 21]},
}


def rot(z: int, offset: int) -> int:
    return (z + offset) % LANE_BITS


def lambda_inputs_full(p: int, x: int, z: int) -> List[Y1Var]:
    if p == 0:
        return [
            (0, x, z),
            (0, (x + 1) % LANES, rot(z, 5)),
            (1, (x + 1) % LANES, rot(z, 5)),
            (2, (x + 1) % LANES, rot(z, 5)),
            (0, (x + 1) % LANES, rot(z, 14)),
            (1, (x + 1) % LANES, rot(z, 14)),
            (2, (x + 1) % LANES, rot(z, 14)),
        ]
    if p == 1:
        return [
            (1, (x + 1) % LANES, z),
            (0, (x + 2) % LANES, rot(z, 5)),
            (1, (x + 2) % LANES, rot(z, 5)),
            (2, (x + 2) % LANES, rot(z, 5)),
            (0, (x + 2) % LANES, rot(z, 14)),
            (1, (x + 2) % LANES, rot(z, 14)),
            (2, (x + 2) % LANES, rot(z, 14)),
        ]
    return [
        (2, x, rot(z, 11)),
        (0, (x + 1) % LANES, rot(z, 16)),
        (1, (x + 1) % LANES, rot(z, 16)),
        (2, (x + 1) % LANES, rot(z, 16)),
        (0, (x + 1) % LANES, rot(z, 25)),
        (1, (x + 1) % LANES, rot(z, 25)),
        (2, (x + 1) % LANES, rot(z, 25)),
    ]


def rhoeast(bit: Y1Var) -> Y1Var:
    p, x, z = bit
    if p == 0:
        return 0, x, z
    if p == 1:
        return 1, x, rot(z, 1)
    return 2, (x + 2) % LANES, rot(z, 8)


def toggle(poly: Polynomial, monomial: Monomial) -> None:
    if monomial in poly:
        poly.remove(monomial)
    else:
        poly.add(monomial)


def y1_chi(bit: Y1Var) -> Polynomial:
    p, x, z = bit
    a = frozenset({(p, x, z)})
    b = frozenset({((p + 1) % PLANES, x, z)})
    d = frozenset({((p + 2) % PLANES, x, z)})
    bd = frozenset(set(b) | set(d))
    return {a, d, bd}


def y2_polynomial(p: int, x: int, z: int) -> Polynomial:
    result: Polynomial = set()
    for source in lambda_inputs_full(p, x, z):
        for monomial in y1_chi(rhoeast(source)):
            toggle(result, monomial)
    return result


def flatten(value: Mapping[int, Mapping[int, Sequence[int]]]) -> List[Y1Var]:
    return [
        (int(p), int(x), int(z))
        for p, lanes in value.items()
        for x, zs in lanes.items()
        for z in zs
    ]


def fixed_y1_values() -> Dict[Y1Var, int]:
    values: Dict[Y1Var, int] = {}
    for source, value in ((Y1_COND_0, 0), (Y1_COND_1, 1)):
        for bit in flatten(source):
            if bit in values and values[bit] != value:
                raise ValueError(f"Conflicting Y1 condition value at {bit}")
            values[bit] = value
    return values


def substitute(poly: Polynomial, fixed: Mapping[Y1Var, int]) -> Polynomial:
    result: Polynomial = set()
    for monomial in poly:
        coefficient = 1
        remaining: Set[Y1Var] = set()
        for variable in monomial:
            if variable in fixed:
                coefficient &= fixed[variable]
            else:
                remaining.add(variable)
        if coefficient:
            toggle(result, frozenset(remaining))
    return result


def edges_from(poly: Polynomial) -> Set[Edge]:
    edges: Set[Edge] = set()
    for monomial in poly:
        if len(monomial) == 2:
            left, right = sorted(monomial)
            edges.add((left, right))
    return edges


def greedy_cover(edges: Set[Edge]) -> Set[Y1Var]:
    remaining = set(edges)
    cover: Set[Y1Var] = set()
    while remaining:
        degree: Dict[Y1Var, int] = {}
        for left, right in remaining:
            degree[left] = degree.get(left, 0) + 1
            degree[right] = degree.get(right, 0) + 1
        chosen = max(sorted(degree), key=lambda item: (degree[item], item))
        cover.add(chosen)
        remaining = {edge for edge in remaining if chosen not in edge}
    return cover


def matching_lower_bound(edges: Set[Edge]) -> int:
    used: Set[Y1Var] = set()
    count = 0
    for left, right in sorted(edges):
        if left not in used and right not in used:
            used.update((left, right))
            count += 1
    return count


def minimum_vertex_cover(edges: Set[Edge]) -> Tuple[Set[Y1Var], int]:
    best = greedy_cover(edges)
    nodes = 0

    def search(remaining: Set[Edge], selected: Set[Y1Var]) -> None:
        nonlocal best, nodes
        nodes += 1
        if not remaining:
            if len(selected) < len(best):
                best = set(selected)
            return
        if len(selected) + matching_lower_bound(remaining) >= len(best):
            return

        degree: Dict[Y1Var, int] = {}
        for left, right in remaining:
            degree[left] = degree.get(left, 0) + 1
            degree[right] = degree.get(right, 0) + 1
        edge = max(
            sorted(remaining),
            key=lambda item: (degree[item[0]] + degree[item[1]], item),
        )
        for chosen in sorted(edge, key=lambda item: (-degree[item], item)):
            reduced = {item for item in remaining if chosen not in item}
            search(reduced, selected | {chosen})

    search(set(edges), set())
    return best, nodes


def fmt(bit: Y1Var) -> str:
    p, x, z = bit
    return f"Y1[{p}][{x}][{z:02d}]"


def main() -> None:
    fixed = fixed_y1_values()
    all_edges: Set[Edge] = set()
    y2_bits = [(0, *bit, 0) for bit in flatten(Y2_COND_0)]
    y2_bits += [(0, *bit, 1) for bit in flatten(Y2_COND_1)]

    for _, p, x, z, _ in y2_bits:
        all_edges.update(edges_from(substitute(y2_polynomial(p, x, z), fixed)))

    cover, search_nodes = minimum_vertex_cover(all_edges)

    print(f"Y1 condition count: {len(fixed)}")
    print(f"Y2 condition count: {len(y2_bits)}")
    print(f"remaining quadratic edges: {len(all_edges)}")
    print(f"minimum additional Y1 guesses: {len(cover)}")
    print(f"vertex-cover search nodes: {search_nodes}")
    print("additional Y1 variables:")
    for bit in sorted(cover):
        print(f"  {fmt(bit)}")


if __name__ == "__main__":
    main()
