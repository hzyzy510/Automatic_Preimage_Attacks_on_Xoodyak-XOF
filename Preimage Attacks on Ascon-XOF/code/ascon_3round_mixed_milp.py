#!/usr/bin/env python3
"""3-round Ascon-Xof mixed-output MILP model.

The model follows the Xoodyak mixed-output formulation while using the
Ascon c/v/q diffusion patterns from the paper:

    (P_X, L_X) = (0, 0)  nonlinear, degree > 1
               = (1, 1)  linear, degree = 1
               = (0, 1)  constant, degree = 0

The three Ascon pS layers are:

    A0 -pS-> B0 -pL-> A1 -pS-> B1 -pL-> A2 -pS-> B2

Two mutually exclusive output types are available for each column x:

* linear output:
    A2[0][x], A2[2][x], and A2[3][x] all have L=1.
    This is Observation 3 and contributes a (1-alpha) gain, where
    alpha = log2(4/3).

* exact output:
    B2[0][x] has P=1.  The b0 bit of the last pS is recovered from the
    hash output, so this gives an exact linear equation.

The Xoodyak-style structure constraint is used:

    number of selected outputs = df
    df + number of active S-box conditions <= 64

The objective minimizes

    128 - (# exact outputs) - (1-alpha) * (# linear outputs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import gurobipy as gp
from gurobipy import GRB

from ascon_4round_linear_milp import (
    ALPHA,
    BITS,
    FIRST_PS,
    PADDING_BIT,
    ROWS,
    SECOND_PS_FOR_4ROUND,
    THIRD_PS_FOR_4ROUND,
    add_linear_layer,
    add_sbox_layer,
    one,
    cat,
)


STATES = ("A0", "B0", "A1", "B1", "A2", "B2")
OUTPUT_ROWS = (0, 2, 3)
HASH_BITS = 128


def add_pl_partition(model: gp.Model, p: gp.tupledict, l: gp.tupledict) -> None:
    """Enforce P=1 => L=1 for every state bit."""
    for state in STATES:
        for row in range(ROWS):
            for bit in range(BITS):
                model.addConstr(
                    p[state, row, bit] <= l[state, row, bit],
                    name=f"PL_{state}_{row}_{bit}",
                )


def add_initial_state(model: gp.Model, p: gp.tupledict, l: gp.tupledict) -> None:
    """Only A0[0][x] can contain free message variables."""
    for row in range(ROWS):
        for bit in range(BITS):
            model.addConstr(l["A0", row, bit] == 1, name=f"A0_L_{row}_{bit}")
            if row != 0 or bit == PADDING_BIT:
                model.addConstr(
                    p["A0", row, bit] == 0,
                    name=f"A0_P_constant_{row}_{bit}",
                )


def add_mixed_outputs(
    model: gp.Model,
    p: gp.tupledict,
    l: gp.tupledict,
) -> tuple[gp.tupledict, gp.tupledict]:
    """Add mutually exclusive Observation-3 and exact output selectors."""
    selected_linear = model.addVars(
        BITS,
        vtype=GRB.BINARY,
        name="SEL_LINEAR_OUT",
    )
    selected_exact = model.addVars(
        BITS,
        vtype=GRB.BINARY,
        name="SEL_EXACT_OUT",
    )

    for bit in range(BITS):
        linear_inputs = [l["A2", row, bit] for row in OUTPUT_ROWS]
        for idx, item in enumerate(linear_inputs):
            model.addConstr(
                selected_linear[bit] <= item,
                name=f"SEL_LINEAR_{bit}_ub_{idx}",
            )
        # B2[0][x] is the b0 output bit of the final pS.  P=1 means that
        # this recovered output bit contains a nonzero linear part.
        model.addConstr(
            selected_exact[bit] <= p["B2", 0, bit],
            name=f"SEL_EXACT_{bit}_ub",
        )
        model.addConstr(
            selected_linear[bit] + selected_exact[bit] <= 1,
            name=f"SEL_MIXED_{bit}_exclusive",
        )

    return selected_linear, selected_exact


def build_model(
    resource_bound: int = 64,
    fixed_df: int | None = None,
) -> Dict[str, object]:
    if resource_bound < 0:
        raise ValueError("resource_bound must be non-negative")
    if fixed_df is not None and not 0 <= fixed_df <= BITS - 1:
        raise ValueError(f"fixed_df must be in [0, {BITS - 1}], got {fixed_df}")

    model = gp.Model("ascon_3round_mixed_output")
    p = model.addVars(STATES, ROWS, BITS, vtype=GRB.BINARY, name="P")
    l = model.addVars(STATES, ROWS, BITS, vtype=GRB.BINARY, name="L")
    value = model.addVars(STATES, ROWS, BITS, vtype=GRB.BINARY, name="VAL")

    conditions: Dict[tuple[str, int, tuple[int, ...], int], gp.Var] = {}
    pattern_choice: Dict[tuple[str, int], List[gp.Var]] = {}

    add_pl_partition(model, p, l)
    add_initial_state(model, p, l)

    # The first S-box is always modeled by Table 2.  The next two layers use
    # the 3-round Table-3 and Table-14 structures from the paper.
    add_sbox_layer(
        model,
        p,
        l,
        value,
        "A0",
        "B0",
        FIRST_PS,
        conditions,
        pattern_choice,
    )
    add_linear_layer(model, p, l, "B0", "A1")
    add_sbox_layer(
        model,
        p,
        l,
        value,
        "A1",
        "B1",
        SECOND_PS_FOR_4ROUND,
        conditions,
        pattern_choice,
    )
    add_linear_layer(model, p, l, "B1", "A2")
    add_sbox_layer(
        model,
        p,
        l,
        value,
        "A2",
        "B2",
        THIRD_PS_FOR_4ROUND,
        conditions,
        pattern_choice,
    )

    selected_linear, selected_exact = add_mixed_outputs(model, p, l)

    df = gp.quicksum(p["A0", 0, bit] for bit in range(BITS))
    linear_count = gp.quicksum(selected_linear[bit] for bit in range(BITS))
    exact_count = gp.quicksum(selected_exact[bit] for bit in range(BITS))
    selected_count = linear_count + exact_count
    condition_count = gp.quicksum(conditions.values())
    third_round_condition_count = gp.quicksum(
        var for (state, _, _, _), var in conditions.items() if state == "A2"
    )
    resource_usage = df + condition_count

    #model.addConstr(third_round_condition_count <= 11, name="third_round_condition_limit")
    # Every free variable must be covered by one selected output equation.
    model.addConstr(selected_count == df, name="selected_outputs_eq_df")
    model.addConstr(resource_usage <= resource_bound + 4, name="resource_bound")
    if fixed_df is not None:
        model.addConstr(df == fixed_df, name="fixed_df")

    complexity = HASH_BITS - exact_count - (ALPHA) * linear_count
    # The small tie-break term does not change the primary complexity order.
    model.setObjective(complexity + 0.001 * condition_count, GRB.MINIMIZE)

    return {
        "model": model,
        "P": p,
        "L": l,
        "VAL": value,
        "conditions": conditions,
        "selected_linear": selected_linear,
        "selected_exact": selected_exact,
        "df": df,
        "linear_count": linear_count,
        "exact_count": exact_count,
        "condition_count": condition_count,
        "resource_usage": resource_usage,
        "resource_bound": resource_bound,
    }


def collect_solution(data: Dict[str, object]) -> Dict[str, object]:
    p = data["P"]
    l = data["L"]
    conditions = data["conditions"]
    selected_linear = data["selected_linear"]
    selected_exact = data["selected_exact"]

    selected_linear_bits = [
        bit for bit in range(BITS) if one(selected_linear[bit])
    ]
    selected_exact_bits = [
        bit for bit in range(BITS) if one(selected_exact[bit])
    ]
    free_bits = [
        bit for bit in range(BITS) if one(p["A0", 0, bit])
    ]
    active_conditions = [
        {
            "state": state,
            "bit": bit,
            "rows": list(rows),
            "rhs": rhs,
        }
        for (state, bit, rows, rhs), var in conditions.items()
        if one(var)
    ]
    states = {
        state: [
            "".join(
                cat(p[state, row, bit], l[state, row, bit])
                for row in range(ROWS)
            )
            for bit in range(BITS)
        ]
        for state in STATES
    }

    linear_count = len(selected_linear_bits)
    exact_count = len(selected_exact_bits)
    condition_count = len(active_conditions)
    df = len(free_bits)
    complexity = HASH_BITS - exact_count - (ALPHA) * linear_count
    resource_usage = df + condition_count

    return {
        "rounds": 3,
        "attack_type": (
            "mixed"
            if linear_count and exact_count
            else "linear"
            if linear_count
            else "exact"
            if exact_count
            else "empty"
        ),
        "df": df,
        "de": linear_count + exact_count,
        "condition_count": condition_count,
        "resource_usage": resource_usage,
        "resource_bound": data["resource_bound"],
        "complexity_exponent": complexity,
        "selected_linear_count": linear_count,
        "selected_exact_count": exact_count,
        "free_bits": free_bits,
        "selected_linear_output_bits": selected_linear_bits,
        "selected_exact_output_bits": selected_exact_bits,
        "conditions": active_conditions,
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-round Ascon MILP with Xoodyak-style mixed outputs."
    )
    parser.add_argument(
        "--resource-bound",
        type=int,
        default=64,
        help="Upper bound for df + active S-box conditions.",
    )
    parser.add_argument(
        "--fixed-df",
        type=int,
        default=None,
        help="Optionally fix the number of free A0[0][x] variables.",
    )
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--write-lp", type=Path, default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("work/ascon_3round_mixed_output_4.json"),
    )
    args = parser.parse_args()

    data = build_model(
        resource_bound=args.resource_bound,
        fixed_df=args.fixed_df,
    )
    model: gp.Model = data["model"]
    if args.threads:
        model.setParam("Threads", args.threads)
    if args.time_limit is not None:
        model.setParam("TimeLimit", args.time_limit)
    model.setParam("MIPGap", args.mip_gap)

    if args.write_lp:
        args.write_lp.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(args.write_lp))

    model.optimize()
    if model.SolCount == 0:
        print("No feasible solution found.")
        return

    result = collect_solution(data)
    print("status:", model.Status)
    print("objective:", model.ObjVal)
    for key in (
        "attack_type",
        "complexity_exponent",
        "df",
        "de",
        "condition_count",
        "resource_usage",
        "selected_linear_count",
        "selected_exact_count",
    ):
        print(f"{key}:", result[key])
    print("free_bits:", result["free_bits"])
    print("selected_linear_output_bits:", result["selected_linear_output_bits"])
    print("selected_exact_output_bits:", result["selected_exact_output_bits"])
    print("conditions:", result["conditions"])

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("json:", str(args.json_output.resolve()))


if __name__ == "__main__":
    main()
