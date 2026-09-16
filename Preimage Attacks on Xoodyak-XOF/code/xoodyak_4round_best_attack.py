#!/usr/bin/env python3
"""Single MILP that mixes exact and linear 4-round Xoodyak attacks.

The model shares the propagation up to Y4, then lets every output position
choose one of two branches:

* linear: select linear bits in Y4[y=2] and use the last-chi approximation.
* exact:  propagate the fourth chi exactly and select linear bits in Z4[y=2].

The same solution may contain both branch types, but one (x,z) output position
cannot be selected as both.  The objective minimizes the mixed complexity
exponent 128 - exact - (1-alpha) * linear.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB


LANES = 4
LANE_BITS = 32
PLANES = 3
DIGEST_BITS = LANES * LANE_BITS
ALPHA = math.log2(4.0 / 3.0)

STATES = ["X1", "Y1", "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3", "X4", "Y4", "Z4"]
COND_STATES = ["Y1", "Y2", "Y3", "Y4"]

GIVEN_FREE_BY_LANE = {
    1: [3, 12, 21],
}

GIVEN_SELECTED_BY_LANE = {
    1: [1, 10, 24],
}

GIVEN_CONDITIONS_EQ_0 = {
    "Y1": {
        0: {0: [5, 10, 19, 28], 1: [1, 10, 24]},
        1: {0: [7, 16, 21, 30]},
        2: {3: [7, 21, 30]},
    },
    "Y2": {
        0: {2: [22, 31]},
        1: {2: [1, 10]},
    },
}

GIVEN_CONDITIONS_EQ_1 = {
    "Y1": {
        0: {3: [7,16,21,30]},
        1: {0: [5, 10, 19, 28], 1: [1, 10, 24]},
        2: {0: [7, 16, 21, 30]},
    },
    "Y2": {
        0: {0: [6, 29], 2: [7, 16, 25]},
        1: {2: [9, 18, 27], 3: [0, 14, 23]},
    },
}


def rot(z: int, offset: int) -> int:
    return (z + offset) % LANE_BITS


def flat(lane: int, z: int) -> int:
    return lane * LANE_BITS + z


def by_lane(bits: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {lane: [] for lane in range(LANES)}
    for lane, z in sorted(bits):
        out[lane].append(z)
    return {lane: zs for lane, zs in out.items() if zs}


def format_state_bits(bits: Sequence[Tuple[str, int, int, int]]) -> Dict[str, Dict[int, Dict[int, List[int]]]]:
    out: Dict[str, Dict[int, Dict[int, List[int]]]] = {}
    for state, p, x, z in sorted(bits):
        out.setdefault(state, {}).setdefault(p, {}).setdefault(x, []).append(z)
    return out


def add_and(model: gp.Model, out: gp.Var, inputs: Sequence[gp.Var], name: str) -> None:
    if not inputs:
        model.addConstr(out == 1, name=f"{name}_empty")
        return
    for idx, var in enumerate(inputs):
        model.addConstr(out <= var, name=f"{name}_ub_{idx}")
    model.addConstr(out >= gp.quicksum(inputs) - len(inputs) + 1, name=f"{name}_lb")


def add_or(model: gp.Model, out: gp.Var, inputs: Sequence[gp.Var], name: str) -> None:
    if not inputs:
        model.addConstr(out == 0, name=f"{name}_empty")
        return
    for idx, var in enumerate(inputs):
        model.addConstr(out >= var, name=f"{name}_lb_{idx}")
    model.addConstr(out <= gp.quicksum(inputs), name=f"{name}_ub")


def add_or_when_linear(
    model: gp.Model,
    out: gp.Var,
    inputs: Sequence[gp.Var],
    lin: gp.Var,
    name: str,
) -> None:
    model.addConstr(out <= lin, name=f"{name}_lin")
    if not inputs:
        model.addConstr(out == 0, name=f"{name}_empty")
        return
    for idx, var in enumerate(inputs):
        model.addConstr(out >= var + lin - 1, name=f"{name}_lb_{idx}")
    model.addConstr(out <= gp.quicksum(inputs), name=f"{name}_ub")


def lambda_inputs_first(plane: int, lane: int, z: int) -> List[Tuple[str, int, int, int]]:
    if plane == 0:
        return [("X1", 2, (lane + 1) % LANES, rot(z, 5)), ("X1", 2, (lane + 1) % LANES, rot(z, 14))]
    if plane == 1:
        return [("X1", 2, (lane + 2) % LANES, rot(z, 5)), ("X1", 2, (lane + 2) % LANES, rot(z, 14))]
    return [
        ("X1", 2, lane, rot(z, 11)),
        ("X1", 2, (lane + 1) % LANES, rot(z, 16)),
        ("X1", 2, (lane + 1) % LANES, rot(z, 25)),
    ]


def lambda_inputs_full(src: str, plane: int, lane: int, z: int) -> List[Tuple[str, int, int, int]]:
    if plane == 0:
        return [
            (src, 0, lane, z),
            (src, 0, (lane + 1) % LANES, rot(z, 5)),
            (src, 1, (lane + 1) % LANES, rot(z, 5)),
            (src, 2, (lane + 1) % LANES, rot(z, 5)),
            (src, 0, (lane + 1) % LANES, rot(z, 14)),
            (src, 1, (lane + 1) % LANES, rot(z, 14)),
            (src, 2, (lane + 1) % LANES, rot(z, 14)),
        ]
    if plane == 1:
        return [
            (src, 1, (lane + 1) % LANES, z),
            (src, 0, (lane + 2) % LANES, rot(z, 5)),
            (src, 1, (lane + 2) % LANES, rot(z, 5)),
            (src, 2, (lane + 2) % LANES, rot(z, 5)),
            (src, 0, (lane + 2) % LANES, rot(z, 14)),
            (src, 1, (lane + 2) % LANES, rot(z, 14)),
            (src, 2, (lane + 2) % LANES, rot(z, 14)),
        ]
    return [
        (src, 2, lane, rot(z, 11)),
        (src, 0, (lane + 1) % LANES, rot(z, 16)),
        (src, 1, (lane + 1) % LANES, rot(z, 16)),
        (src, 2, (lane + 1) % LANES, rot(z, 16)),
        (src, 0, (lane + 1) % LANES, rot(z, 25)),
        (src, 1, (lane + 1) % LANES, rot(z, 25)),
        (src, 2, (lane + 1) % LANES, rot(z, 25)),
    ]


def rhoeast_input(src: str, plane: int, lane: int, z: int) -> Tuple[str, int, int, int]:
    if plane == 0:
        return src, 0, lane, z
    if plane == 1:
        return src, 1, lane, rot(z, 1)
    return src, 2, (lane + 2) % LANES, rot(z, 8)


def parse_list(text: str | None) -> List[int] | None:
    if text is None:
        return None
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("Expected a JSON list of integers.")
    return value


def lane_dict_to_pairs(value: Dict[int, Sequence[int]]) -> List[Tuple[int, int]]:
    return [(lane, z) for lane, zs in value.items() for z in zs]


def condition_dict_to_bits(value: Dict[str, Dict[int, Dict[int, Sequence[int]]]]) -> List[Tuple[str, int, int, int]]:
    return [(state, p, x, z) for state, planes in value.items() for p, lanes in planes.items() for x, zs in lanes.items() for z in zs]


def given_start_solution() -> Dict[str, object]:
    guessed_zero = lane_dict_to_pairs(GIVEN_FREE_BY_LANE)
    all_input_bits = {(x, z) for x in range(LANES) for z in range(LANE_BITS)}
    guessed_zero_set = set(guessed_zero)
    if not guessed_zero_set <= all_input_bits:
        extra = sorted(guessed_zero_set - all_input_bits)
        raise ValueError(f"given free variables are out of range: {extra}")
    guessed_one = sorted(all_input_bits - guessed_zero_set)
    return {
        "path": "embedded user-provided 4-round start",
        "guessed_one": guessed_one,
        "guessed_zero": guessed_zero,
        "conditions0": condition_dict_to_bits(GIVEN_CONDITIONS_EQ_0),
        "conditions1": condition_dict_to_bits(GIVEN_CONDITIONS_EQ_1),
        "selected": lane_dict_to_pairs(GIVEN_SELECTED_BY_LANE),
        "selected_linear": [],
        "selected_exact": [],
    }


def apply_start_solution(data: Dict[str, object], start: Dict[str, object]) -> None:
    guessed = data["guessed"]
    cond0 = data["cond0"]
    cond1 = data["cond1"]
    lin = data["lin"]
    part = data["part"]
    selected_linear = data["selected_linear"]
    selected_exact = data["selected_exact"]

    guessed_one = set(start["guessed_one"])
    guessed_zero = set(start["guessed_zero"])
    conditions0 = set(start["conditions0"])
    conditions1 = set(start["conditions1"])
    selected_all = set(start.get("selected", []))
    linear_sel = set(start.get("selected_linear", []))
    exact_sel = set(start.get("selected_exact", []))
    has_split = bool(linear_sel or exact_sel)
    cond_bits = conditions0 | conditions1

    for x in range(LANES):
        for z in range(LANE_BITS):
            if (x, z) in guessed_one:
                guessed[x, z].Start = 1
            elif (x, z) in guessed_zero:
                guessed[x, z].Start = 0
            if has_split:
                selected_linear[x, z].Start = 1 if (x, z) in linear_sel else 0
                selected_exact[x, z].Start = 1 if (x, z) in exact_sel else 0
            elif selected_all:
                if (x, z) not in selected_all:
                    selected_linear[x, z].Start = 0
                    selected_exact[x, z].Start = 0

    for state in COND_STATES:
        for p in range(PLANES):
            for x in range(LANES):
                for z in range(LANE_BITS):
                    if (state, p, x, z) in cond_bits:
                        lin[state, p, x, z].Start = 1
                        part[state, p, x, z].Start = 0
                    cond0[state, p, x, z].Start = 1 if (state, p, x, z) in conditions0 else 0
                    cond1[state, p, x, z].Start = 1 if (state, p, x, z) in conditions1 else 0


def build_model(
    resource_bound: int,
    min_conditions: int,
    fixed_guessed: Sequence[int] | None,
    fixed_selected: Sequence[int] | None,
) -> Dict[str, object]:
    model = gp.Model("xoodyak_4round_best_exact_or_linear")
    lin = model.addVars(STATES, PLANES, LANES, LANE_BITS, vtype=GRB.BINARY, name="LIN")
    part = model.addVars(STATES, PLANES, LANES, LANE_BITS, vtype=GRB.BINARY, name="PART")
    cond0 = model.addVars(COND_STATES, PLANES, LANES, LANE_BITS, vtype=GRB.BINARY, name="COND_EQ_0")
    cond1 = model.addVars(COND_STATES, PLANES, LANES, LANE_BITS, vtype=GRB.BINARY, name="COND_EQ_1")
    selected_linear = model.addVars(LANES, LANE_BITS, vtype=GRB.BINARY, name="SEL_LINEAR_Y4")
    selected_exact = model.addVars(LANES, LANE_BITS, vtype=GRB.BINARY, name="SEL_EXACT_Z4")
    guessed = model.addVars(LANES, LANE_BITS, vtype=GRB.BINARY, name="G_INPUT_CONST")
    for state in STATES:
        for p in range(PLANES):
            for x in range(LANES):
                for z in range(LANE_BITS):
                    model.addConstr(part[state, p, x, z] <= lin[state, p, x, z], name=f"part_implies_lin_{state}_{p}_{x}_{z}")

    for p in range(PLANES):
        for x in range(LANES):
            for z in range(LANE_BITS):
                model.addConstr(lin["X1", p, x, z] == 1, name=f"x1_lin_{p}_{x}_{z}")
                if p == 2:
                    model.addConstr(part["X1", p, x, z] + guessed[x, z] == 1, name=f"x1_part_or_guess_{x}_{z}")
                else:
                    model.addConstr(part["X1", p, x, z] == 0, name=f"x1_const_plane_{p}_{x}_{z}")

    def xor_layer(dst: str, inputs_fn) -> None:
        for p in range(PLANES):
            for x in range(LANES):
                for z in range(LANE_BITS):
                    inputs = inputs_fn(p, x, z)
                    add_and(model, lin[dst, p, x, z], [lin[item] for item in inputs], f"lin_{dst}_{p}_{x}_{z}")
                    add_or_when_linear(
                        model,
                        part[dst, p, x, z],
                        [part[item] for item in inputs],
                        lin[dst, p, x, z],
                        f"part_{dst}_{p}_{x}_{z}",
                    )

    def copy_layer(src: str, dst: str) -> None:
        for p in range(PLANES):
            for x in range(LANES):
                for z in range(LANE_BITS):
                    source = rhoeast_input(src, p, x, z)
                    model.addConstr(lin[dst, p, x, z] == lin[source], name=f"copy_lin_{dst}_{p}_{x}_{z}")
                    model.addConstr(part[dst, p, x, z] == part[source], name=f"copy_part_{dst}_{p}_{x}_{z}")

    def known0(state: str, p: int, x: int, z: int):
        return cond0[state, p, x, z] if state in COND_STATES else 0

    def known1(state: str, p: int, x: int, z: int):
        return cond1[state, p, x, z] if state in COND_STATES else 0

    def chi_layer(src: str, dst: str) -> None:
        for p in range(PLANES):
            for x in range(LANES):
                for z in range(LANE_BITS):
                    a = (src, p, x, z)
                    b = (src, (p + 1) % PLANES, x, z)
                    d = (src, (p + 2) % PLANES, x, z)

                    b_const = model.addVar(vtype=GRB.BINARY, name=f"b_const_{dst}_{p}_{x}_{z}")
                    d_const = model.addVar(vtype=GRB.BINARY, name=f"d_const_{dst}_{p}_{x}_{z}")
                    can_linear = model.addVar(vtype=GRB.BINARY, name=f"can_linear_{dst}_{p}_{x}_{z}")
                    kill_product = model.addVar(vtype=GRB.BINARY, name=f"kill_product_{dst}_{p}_{x}_{z}")
                    normal_product_linear = model.addVar(vtype=GRB.BINARY, name=f"normal_product_linear_{dst}_{p}_{x}_{z}")
                    product_linear = model.addVar(vtype=GRB.BINARY, name=f"product_linear_{dst}_{p}_{x}_{z}")
                    d_prop = model.addVar(vtype=GRB.BINARY, name=f"d_prop_{dst}_{p}_{x}_{z}")
                    b_prop = model.addVar(vtype=GRB.BINARY, name=f"b_prop_{dst}_{p}_{x}_{z}")

                    model.addConstr(b_const + part[b] == 1, name=f"b_const_def_{dst}_{p}_{x}_{z}")
                    model.addConstr(d_const + part[d] == 1, name=f"d_const_def_{dst}_{p}_{x}_{z}")
                    add_or(model, can_linear, [b_const, d_const], f"can_linear_or_{dst}_{p}_{x}_{z}")
                    add_or(model, kill_product, [known1(*b), known0(*d)], f"kill_product_or_{dst}_{p}_{x}_{z}")
                    add_and(
                        model,
                        normal_product_linear,
                        [lin[b], lin[d], can_linear],
                        f"normal_product_linear_{dst}_{p}_{x}_{z}",
                    )
                    add_or(
                        model,
                        product_linear,
                        [kill_product, normal_product_linear],
                        f"product_linear_or_{dst}_{p}_{x}_{z}",
                    )
                    add_and(model, lin[dst, p, x, z], [lin[a], product_linear], f"chi_lin_{dst}_{p}_{x}_{z}")

                    model.addConstr(d_prop <= part[d], name=f"d_prop_pd_{dst}_{p}_{x}_{z}")
                    model.addConstr(d_prop <= b_const, name=f"d_prop_bc_{dst}_{p}_{x}_{z}")
                    model.addConstr(d_prop <= 1 - known1(*b), name=f"d_prop_not_b1_{dst}_{p}_{x}_{z}")
                    model.addConstr(d_prop >= part[d] + b_const + (1 - known1(*b)) - 2, name=f"d_prop_lb_{dst}_{p}_{x}_{z}")

                    model.addConstr(b_prop <= part[b], name=f"b_prop_pb_{dst}_{p}_{x}_{z}")
                    model.addConstr(b_prop <= d_const, name=f"b_prop_dc_{dst}_{p}_{x}_{z}")
                    model.addConstr(b_prop <= 1 - known0(*d), name=f"b_prop_not_d0_{dst}_{p}_{x}_{z}")
                    model.addConstr(b_prop >= part[b] + d_const + (1 - known0(*d)) - 2, name=f"b_prop_lb_{dst}_{p}_{x}_{z}")

                    add_or_when_linear(
                        model,
                        part[dst, p, x, z],
                        [part[a], d_prop, b_prop],
                        lin[dst, p, x, z],
                        f"chi_part_{dst}_{p}_{x}_{z}",
                    )

    xor_layer("Y1", lambda p, x, z: lambda_inputs_first(p, x, z))

    for state in COND_STATES:
        for p in range(PLANES):
            for x in range(LANES):
                for z in range(LANE_BITS):
                    model.addConstr(cond0[state, p, x, z] + cond1[state, p, x, z] <= 1, name=f"cond_exclusive_{state}_{p}_{x}_{z}")
                    model.addConstr(cond0[state, p, x, z] <= lin[state, p, x, z], name=f"cond0_lin_{state}_{p}_{x}_{z}")
                    model.addConstr(cond1[state, p, x, z] <= lin[state, p, x, z], name=f"cond1_lin_{state}_{p}_{x}_{z}")
                    model.addConstr(cond0[state, p, x, z] <= 1 - part[state, p, x, z], name=f"cond0_gray_{state}_{p}_{x}_{z}")
                    model.addConstr(cond1[state, p, x, z] <= 1 - part[state, p, x, z], name=f"cond1_gray_{state}_{p}_{x}_{z}")
    chi_layer("Y1", "Z1")
    copy_layer("Z1", "X2")
    xor_layer("Y2", lambda p, x, z: lambda_inputs_full("X2", p, x, z))
    chi_layer("Y2", "Z2")
    copy_layer("Z2", "X3")
    xor_layer("Y3", lambda p, x, z: lambda_inputs_full("X3", p, x, z))
    chi_layer("Y3", "Z3")
    copy_layer("Z3", "X4")
    xor_layer("Y4", lambda p, x, z: lambda_inputs_full("X4", p, x, z))
    chi_layer("Y4", "Z4")

    free_count = gp.quicksum(part["X1", 2, x, z] for x in range(LANES) for z in range(LANE_BITS))
    selected_linear_count = gp.quicksum(selected_linear[x, z] for x in range(LANES) for z in range(LANE_BITS))
    selected_exact_count = gp.quicksum(selected_exact[x, z] for x in range(LANES) for z in range(LANE_BITS))
    selected_count = selected_linear_count + selected_exact_count
    y1_conditions = gp.quicksum(cond0["Y1", p, x, z] + cond1["Y1", p, x, z] for p in range(PLANES) for x in range(LANES) for z in range(LANE_BITS))
    y2_conditions = gp.quicksum(cond0["Y2", p, x, z] + cond1["Y2", p, x, z] for p in range(PLANES) for x in range(LANES) for z in range(LANE_BITS))
    y3_conditions = gp.quicksum(cond0["Y3", p, x, z] + cond1["Y3", p, x, z] for p in range(PLANES) for x in range(LANES) for z in range(LANE_BITS))
    y4_conditions = gp.quicksum(cond0["Y4", p, x, z] + cond1["Y4", p, x, z] for p in range(PLANES) for x in range(LANES) for z in range(LANE_BITS))
    weighted_conditions = y1_conditions + 2.6 * y2_conditions + 32 * y3_conditions + 128 * y4_conditions
    condition_count = y1_conditions + y2_conditions + y3_conditions + y4_conditions

    for x in range(LANES):
        for z in range(LANE_BITS):
            model.addConstr(selected_linear[x, z] <= lin["Y4", 2, x, z], name=f"sel_linear_y4_{x}_{z}")
            model.addConstr(selected_exact[x, z] <= lin["Z4", 2, x, z], name=f"sel_exact_z4_{x}_{z}")
            model.addConstr(selected_linear[x, z] + selected_exact[x, z] <= 1, name=f"sel_exact_linear_exclusive_{x}_{z}")

    model.addConstr(selected_count == free_count, name="selected_equals_free")
    model.addConstr(free_count + weighted_conditions <= resource_bound, name="resource_bound")
    if min_conditions > 0:
        model.addConstr(condition_count >= min_conditions, name="min_conditions")

    if fixed_guessed is not None:
        guessed_set = set(fixed_guessed)
        for x in range(LANES):
            for z in range(LANE_BITS):
                bit = flat(x, z)
                model.addConstr(guessed[x, z] == (1 if bit in guessed_set else 0), name=f"fixed_g_{bit}")

    if fixed_selected is not None:
        selected_set = set(fixed_selected)
        for x in range(LANES):
            for z in range(LANE_BITS):
                bit = flat(x, z)
                model.addConstr(
                    selected_linear[x, z] + selected_exact[x, z] == (1 if bit in selected_set else 0),
                    name=f"fixed_sel_{bit}",
                )

    complexity = DIGEST_BITS - selected_exact_count - (1.0 - ALPHA) * selected_linear_count
    model.setObjective(complexity + 0.001 * weighted_conditions, GRB.MINIMIZE)

    return {
        "model": model,
        "lin": lin,
        "part": part,
        "cond0": cond0,
        "cond1": cond1,
        "selected_linear": selected_linear,
        "selected_exact": selected_exact,
        "guessed": guessed,
    }


def bit(var: gp.Var) -> int:
    return 1 if var.X > 0.5 else 0


def collect_solution(data: Dict[str, object]) -> Dict[str, object]:
    lin = data["lin"]
    part = data["part"]
    cond0 = data["cond0"]
    cond1 = data["cond1"]
    selected_linear = data["selected_linear"]
    selected_exact = data["selected_exact"]
    guessed = data["guessed"]

    bad_conditions = [
        (s, p, x, z)
        for s in COND_STATES
        for p in range(PLANES)
        for x in range(LANES)
        for z in range(LANE_BITS)
        if (bit(cond0[s, p, x, z]) or bit(cond1[s, p, x, z]))
        and (not bit(lin[s, p, x, z]) or bit(part[s, p, x, z]))
    ]
    if bad_conditions:
        raise RuntimeError(f"conditions on non-gray cells: {format_state_bits(bad_conditions)}")

    guessed_bits = [(x, z) for x in range(LANES) for z in range(LANE_BITS) if bit(guessed[x, z])]
    free_bits = [(x, z) for x in range(LANES) for z in range(LANE_BITS) if bit(part["X1", 2, x, z])]
    selected_linear_bits = [(x, z) for x in range(LANES) for z in range(LANE_BITS) if bit(selected_linear[x, z])]
    selected_exact_bits = [(x, z) for x in range(LANES) for z in range(LANE_BITS) if bit(selected_exact[x, z])]
    selected_bits = sorted(selected_linear_bits + selected_exact_bits)
    final_linear_y4 = [(x, z) for x in range(LANES) for z in range(LANE_BITS) if bit(lin["Y4", 2, x, z])]
    final_linear_z4 = [(x, z) for x in range(LANES) for z in range(LANE_BITS) if bit(lin["Z4", 2, x, z])]
    conditions0 = [(s, p, x, z) for s in COND_STATES for p in range(PLANES) for x in range(LANES) for z in range(LANE_BITS) if bit(cond0[s, p, x, z])]
    conditions1 = [(s, p, x, z) for s in COND_STATES for p in range(PLANES) for x in range(LANES) for z in range(LANE_BITS) if bit(cond1[s, p, x, z])]
    weights = {"Y1": 1, "Y2": 2.6, "Y3": 32, "Y4": 128}
    weighted = sum(weights[s] for s, _, _, _ in conditions0 + conditions1)
    if selected_linear_bits and selected_exact_bits:
        attack_type = "mixed"
    elif selected_exact_bits:
        attack_type = "exact"
    elif selected_linear_bits:
        attack_type = "linear"
    else:
        attack_type = "empty"
    complexity = DIGEST_BITS - len(selected_exact_bits) - (1.0 - ALPHA) * len(selected_linear_bits)

    return {
        "attack_type": attack_type,
        "complexity_exponent": complexity,
        "guessed_count": len(guessed_bits),
        "free_variable_count": len(free_bits),
        "condition_count": len(conditions0) + len(conditions1),
        "weighted_condition_count": weighted,
        "resource_usage": len(free_bits) + weighted,
        "selected_count": len(selected_bits),
        "selected_linear_count": len(selected_linear_bits),
        "selected_exact_count": len(selected_exact_bits),
        "final_linearizable_y4_count": len(final_linear_y4),
        "final_linearizable_z4_count": len(final_linear_z4),
        "guessed_by_lane": by_lane(guessed_bits),
        "free_variables_by_lane": by_lane(free_bits),
        "selected_by_lane": by_lane(selected_bits),
        "selected_linear_by_lane": by_lane(selected_linear_bits),
        "selected_exact_by_lane": by_lane(selected_exact_bits),
        "final_linearizable_y4_by_lane": by_lane(final_linear_y4),
        "final_linearizable_z4_by_lane": by_lane(final_linear_z4),
        "conditions_eq_0": format_state_bits(conditions0),
        "conditions_eq_1": format_state_bits(conditions1),
    }


def write_mixed_svg(path: Path, data: Dict[str, object], result: Dict[str, object]) -> None:
    lin = data["lin"]
    part = data["part"]
    cond0 = data["cond0"]
    cond1 = data["cond1"]
    selected_linear = data["selected_linear"]
    selected_exact = data["selected_exact"]

    cell = 7
    row_gap = 1
    panel_gap = 28
    left = 68
    top = 38
    grid_w = LANE_BITS * cell
    grid_h = PLANES * LANES * cell
    width = left + grid_w + 34
    height = top + len(STATES) * (grid_h + panel_gap) + 58

    def color(state: str, p: int, x: int, z: int) -> str:
        if state in COND_STATES:
            if bit(cond0[state, p, x, z]):
                return "#d73a49"
            if bit(cond1[state, p, x, z]):
                return "#2da44e"
        if state == "Y4" and p == 2 and bit(selected_linear[x, z]):
            return "#8ec5ff"
        if state == "Z4" and p == 2 and bit(selected_exact[x, z]):
            return "#b197fc"
        if not bit(lin[state, p, x, z]):
            return "#ffffff"
        return "#ffd84d" if bit(part[state, p, x, z]) else "#b8b8b8"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Consolas,Menlo,monospace;font-size:11px;fill:#222}.title{font-size:14px;font-weight:700}.small{font-size:9px}</style>',
        f'<text x="20" y="22" class="title">Xoodyak mixed 4-round attack ({result["attack_type"]})</text>',
        '<rect x="20" y="34" width="11" height="11" fill="#ffd84d" stroke="#666"/><text x="37" y="43">linear variable</text>',
        '<rect x="140" y="34" width="11" height="11" fill="#d73a49" stroke="#666"/><text x="157" y="43">condition = 0</text>',
        '<rect x="262" y="34" width="11" height="11" fill="#2da44e" stroke="#666"/><text x="279" y="43">condition = 1</text>',
        '<rect x="384" y="34" width="11" height="11" fill="#b8b8b8" stroke="#666"/><text x="401" y="43">constant</text>',
        '<rect x="486" y="34" width="11" height="11" fill="#8ec5ff" stroke="#666"/><text x="503" y="43">linear sel</text>',
        '<rect x="590" y="34" width="11" height="11" fill="#b197fc" stroke="#666"/><text x="607" y="43">exact sel</text>',
    ]

    y0 = top + 40
    for idx, state in enumerate(STATES):
        panel_y = y0 + idx * (grid_h + panel_gap)
        lines.append(f'<text x="20" y="{panel_y - 8}" class="title">{state}</text>')
        for z in range(0, LANE_BITS + 1, 8):
            xpos = left + z * cell
            lines.append(f'<text x="{xpos - 3}" y="{panel_y - 1}" class="small">{z}</text>')
        for p in range(PLANES):
            for x in range(LANES):
                row_y = panel_y + (p * LANES + x) * cell
                lines.append(f'<text x="24" y="{row_y + cell - 1}" class="small">y{p}x{x}</text>')
                for z in range(LANE_BITS):
                    xpos = left + z * cell
                    lines.append(
                        f'<rect x="{xpos}" y="{row_y}" width="{cell - row_gap}" height="{cell - row_gap}" '
                        f'fill="{color(state, p, x, z)}" stroke="#444" stroke-width="0.2"/>'
                    )
        for z in range(0, LANE_BITS + 1, 8):
            xpos = left + z * cell
            lines.append(f'<line x1="{xpos}" y1="{panel_y}" x2="{xpos}" y2="{panel_y + grid_h}" stroke="#222" stroke-width="0.45" opacity="0.4"/>')

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single Xoodyak 4-round MILP mixing exact and linear output bits.")
    parser.add_argument("--resource-bound", type=int, default=128)
    parser.add_argument("--min-conditions", type=int, default=0)
    parser.add_argument("--fixed-guessed", type=str, default=None, help="JSON flat list of guessed X1[y=2] bits.")
    parser.add_argument("--fixed-selected", type=str, default=None, help="JSON flat list of selected output bits.")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--svg-output", type=Path, default=Path("work/xoodyak_4round_best_attack_1_2.6_32_128.svg"))
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    data = build_model(
        resource_bound=args.resource_bound,
        min_conditions=args.min_conditions,
        fixed_guessed=parse_list(args.fixed_guessed),
        fixed_selected=parse_list(args.fixed_selected),
    )
    model: gp.Model = data["model"]
    model.setParam("Threads", 12)
    model.setParam("MIPFocus", 2)
    model.setParam("Presolve", 2)
    model.setParam("Cuts", 2)
    model.setParam("Heuristics", 0.1)
    model.setParam("MIPGap", args.mip_gap)
    if args.seed is not None:
        model.setParam("Seed", args.seed)
    if args.time_limit is not None:
        model.setParam("TimeLimit", args.time_limit)
    if args.log_file is not None:
        model.setParam("LogFile", args.log_file)

    # start = given_start_solution()
    # apply_start_solution(data, start)
    model.optimize()
    if model.SolCount == 0:
        print("No feasible solution found.")
        return

    result = collect_solution(data)
    print("status:", model.Status)
    print("objective:", model.ObjVal)
    for key in [
        "attack_type",
        "complexity_exponent",
        "guessed_count",
        "free_variable_count",
        "condition_count",
        "weighted_condition_count",
        "resource_usage",
        "selected_count",
        "selected_linear_count",
        "selected_exact_count",
        "final_linearizable_y4_count",
        "final_linearizable_z4_count",
    ]:
        print(f"{key}:", result[key])
    print()
    print("guessed constants by lane:")
    print(result["guessed_by_lane"])
    print()
    print("free variables by lane:")
    print(result["free_variables_by_lane"])
    print()
    print("selected outputs by lane:")
    print(result["selected_by_lane"])
    print()
    print("conditions = 0:")
    print(result["conditions_eq_0"])
    print("conditions = 1:")
    print(result["conditions_eq_1"])

    write_mixed_svg(args.svg_output, data, result)
    print()
    print("winner SVG:")
    print(str(args.svg_output.resolve()))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
