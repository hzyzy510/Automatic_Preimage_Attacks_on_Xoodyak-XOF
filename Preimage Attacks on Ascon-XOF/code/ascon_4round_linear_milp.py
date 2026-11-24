#!/usr/bin/env python3
"""4-round Ascon-Xof linear-output MILP model.

This script is the Ascon counterpart of the local Xoodyak MILP code.  It uses
the c/v/q S-box propagation tables from "Automatic Preimage Attack Framework
on Ascon" and encodes every category with the variables requested by the user:

    (P_X, L_X) = (0, 0)  nonlinear, degree > 1
               = (1, 1)  linear, degree = 1
               = (0, 1)  constant, degree = 0

So L_X = 1 means degree at most one, and P_X = 1 means a nonzero linear part.

The model searches the 3-round structure used by a 4-round preimage attack:

    A0 -pS-> B0 -pL-> A1 -pS-> B1 -pL-> A2 -pS-> B2 -pL-> A3

The selected hash equations are the "linear output" positions where
A3[0][x], A3[2][x], and A3[3][x] all have L = 1, matching Observation 3.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB


ROWS = 5
BITS = 64
PADDING_BIT = 63
ALPHA = 0.585
ROT_OFFSETS = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))

STATES = ("A0", "B0", "A1", "B1", "A2", "B2", "A3")


class Pattern(NamedTuple):
    inputs: str
    outputs: str
    conditions: Tuple[str, ...] = ()


def parse_pattern_table(text: str) -> List[Pattern]:
    patterns: List[Pattern] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed pattern line: {raw_line!r}")
        conditions: Tuple[str, ...] = ()
        if len(fields) >= 3 and fields[2] != "-":
            conditions = tuple(fields[2].split(";"))
        inputs, outputs = fields[0], fields[1]
        if len(inputs) != ROWS or len(outputs) != ROWS:
            raise ValueError(f"Bad pattern width: {raw_line!r}")
        patterns.append(Pattern(inputs, outputs, conditions))
    return patterns


# Paper Table 2: first pS.
FIRST_PS = parse_pattern_table(
    """
    ccccc ccccc -
    vcccc vvccc a1=0;a3+a4=1
    vcccc vvcvc a1=0;a3+a4=0
    vcccc cvccv a1=1;a3+a4=1
    vcccc cvcvv a1=1;a3+a4=0
    """
)


# Paper Table 3, rows used as the second pS of 3-round structures.
SECOND_PS_FOR_4ROUND = parse_pattern_table(
    """
    ccccc ccccc -
    vcccc vvccc a1=0;a3+a4=1
    vcccc cvccv a1=1;a3+a4=1
    cvccc ccvvc a2=0;a3=1;a0+a4=1
    ccccv cvccv a0=1;a1=0;a3=1
    ccccv vvccc a0=1;a1=1;a3=1
    vvccc qvvvq -
    cvccv qvvvq -
    """
)


# Paper Table 14: third pS. Only rows relevant to the appendix table are listed.
THIRD_PS_FOR_4ROUND = parse_pattern_table(
    """
    ccccc vvvvv -
    vcccc vvvvv -
    cvccc vvvvv -
    cccvc vvvvv -
    ccccv vvvvv -
    vvccc qvvvq -
    cvccv qvvvq -
    vccvc vvvqv -
    cvcvc vqvvv -
    vvcvc qqvqq -
    vcccv vvvqv -
    vvccv qvvqq -
    cccvv vvqvv -
    vccvv vvqqv -
    cvcvv qqqvq -
    vvcvv qqqqq -
    ccvcc vvvvv -
    ccvcv vvvvv -
    ccvvc vqvvv -
    ccvvv vqqvv -
    cvvcc qqvvv -
    cvvcv qqvvq -
    cvvvc qqvvv -
    cvvvv qqqvq -
    vcvcc vvvvv -
    vcvcv vvvqv -
    vcvvc vqvqv -
    vcvvv vqqqv -
    vvvcc qqvvq -
    vvvcv qqvqq -
    vvvvc qqvqq -
    vvvvv qqqqq -
    qvccc qqvqq -
    qvccv qqvqq -
    qvcvc qqvqq -
    qvcvv qqqqq -
    qvvcc qqvqq -
    qvvcv qqvqq -
    qvvvc qqvqq -
    qvvvv qqqqq -
    qcccc qqvqv a1=0
    qcccc qqvvv a1=0;a3+a4=1
    qcccc vqvqq a1=1
    qcccc vqvvq a1=1;a3+a4=1
    qcccv qqvqv a1=0
    qcccv vqvqq a1=1
    qccvc qqvqv a1=0
    qccvc vqvqq a1=1
    qccvv qqqqv a1=0
    qccvv vqqqq a1=1
    qcvcc qqvqv a1=0
    qcvcc qqvvv a1=0;a3+a4=1
    qcvcc vqvqq a1=1
    qcvcc vqvvq a1=1;a3+a4=1
    qcvcv qqvqv a1=0
    qcvcv vqvqq a1=1
    qcvvc qqvqv a1=0
    qcvvc vqvqq a1=1
    qcvvv vqqqq a1=1
    qcvvv qqqqv a1=0
    qvvcc qqvvq a3+a4=1
    qvccc qqvvq a3+a4=1
    ccccq qqqqv a1=1
    ccccq qqqvv a1=1;a0=1
    ccccq qqvqv a1=1;a3=1
    ccccq qqvvv a1=1;a0=1;a3=1
    ccccq vqqqq a1=0
    ccccq vqqvq a1=0;a0=1
    ccccq vqvqq a1=0;a3=1
    ccccq vqvvq a1=0;a3=1;a0=1
    cccvq qqqqv a1=1
    cccvq qqqvv a1=1;a0=1
    cccvq vqqqq a1=0
    cccvq vqqvq a1=0;a0=1
    ccvcq qqqqv a1=1
    ccvcq qqqvv a1=1;a0=1
    ccvcq qqvqv a1=1;a3=1
    ccvcq qqvvv a1=1;a0=1;a3=1
    ccvcq vqqqq a1=0
    ccvcq vqqvq a1=0;a0=1
    ccvcq vqvqq a1=0;a3=1
    ccvcq vqvvq a1=0;a3=1;a0=1
    ccvvq qqqqv a1=1
    ccvvq qqqvv a1=1;a0=1
    ccvvq vqqqq a1=0
    ccvvq vqqvq a1=0;a0=1
    cvccq qqqvq a0=1
    cvccq qqvqq a3=1
    cvccq qqvvq a0=1;a3=1
    cvcvq qqqvq a0=1
    cvvcq qqqvq a0=1
    cvvcq qqvqq a3=1
    cvvcq qqvvq a3=1;a0=1
    cvvvq qqqvq a0=1
    vcccq qqqqv a1=1
    vcccq qqvqq a3=1
    vcccq qqvqv a1=1;a3=1
    vccvq qqqqv a1=1
    vccvq qqvqq a3=1
    vcvcq qqqqv a1=1
    vcvcq qqvqq a3=1
    vcvcq qqvqv a1=1;a3=1
    vcvvq qqqqv a1=1
    vvccq qqvqq a3=1
    vvvcq qqvqq a3=1
    cvccq qqqqq -
    cvcvq qqqqq -
    cvvcq qqqqq -
    cvvvq qqqqq -
    vcccq qqqqq -
    vccvq qqqqq -
    vcvcq qqqqq -
    vcvvq qqqqq -
    vvccq qqqqq -
    vvcvq qqqqq -
    vvvvq qqqqq -
    qcccq qqqqq -
    qccvq qqqqq -
    qcvcq qqqqq -
    qcvvq qqqqq -
    qvccq qqqqq -
    qvcvq qqqqq -
    qvvcq qqqqq -
    qvvvq qqqqq -
    vvvcq qqqqq -
    """
)


def category_pl(kind: str) -> Tuple[int, int]:
    if kind == "q":
        return 0, 0
    if kind == "v":
        return 1, 1
    if kind == "c":
        return 0, 1
    raise ValueError(kind)


def add_or(model: gp.Model, out: gp.Var, inputs: Sequence[gp.Var], name: str) -> None:
    for idx, item in enumerate(inputs):
        model.addConstr(out >= item, name=f"{name}_lb_{idx}")
    model.addConstr(out <= gp.quicksum(inputs), name=f"{name}_ub")


def add_and(model: gp.Model, out: gp.Var, inputs: Sequence[gp.Var], name: str) -> None:
    for idx, item in enumerate(inputs):
        model.addConstr(out <= item, name=f"{name}_ub_{idx}")
    model.addConstr(out >= gp.quicksum(inputs) - len(inputs) + 1, name=f"{name}_lb")


def condition_rows(condition: str) -> Tuple[int, ...]:
    lhs, _ = condition.split("=")
    return tuple(sorted(int(term[1:]) for term in lhs.split("+")))


def condition_rhs(condition: str) -> int:
    return int(condition.split("=")[1])


def condition_tag(condition: str) -> str:
    """Return a Gurobi-safe tag for a condition such as ``a3+a4=1``."""
    return condition.replace("+", "_plus_").replace("=", "_eq_")


def condition_key(state: str, bit: int, condition: str) -> Tuple[str, int, Tuple[int, ...], int]:
    return state, bit, condition_rows(condition), condition_rhs(condition)


def add_condition(
    model: gp.Model,
    p: gp.tupledict,
    l: gp.tupledict,
    value: gp.tupledict,
    key: Tuple[str, int, Tuple[int, ...], int],
    conditions: Dict[Tuple[str, int, Tuple[int, ...], int], gp.Var],
) -> gp.Var:
    if key in conditions:
        return conditions[key]

    state, bit, rows, rhs = key
    name = f"{state}_x{bit}_{'_'.join(map(str, rows))}_eq_{rhs}"
    cond = model.addVar(vtype=GRB.BINARY, name=f"COND_{name}")
    conditions[key] = cond

    for row in rows:
        # A condition from the paper is imposed on a constant input bit.
        # In the requested encoding, constant means (P,L)=(0,1).
        model.addConstr(cond <= l[state, row, bit], name=f"{name}_l_{row}")
        model.addConstr(cond <= 1 - p[state, row, bit], name=f"{name}_p_{row}")

    # All conditions used in the paper are either a_i=b or a_i+a_j=b.
    # For two variables over GF(2), a_i+a_j=1 is exactly integer sum == 1.
    # Since every input is binary, rhs=0 and rhs=1 are both represented by
    # the same equality with an integer sum.
    lhs = gp.quicksum(value[state, row, bit] for row in rows)
    big_m = len(rows) + 1
    model.addConstr(lhs - rhs <= big_m * (1 - cond), name=f"{name}_ub")
    model.addConstr(lhs - rhs >= -big_m * (1 - cond), name=f"{name}_lb")
    return cond


def add_pl_partition(model: gp.Model, p: gp.tupledict, l: gp.tupledict) -> None:
    for state in STATES:
        for row in range(ROWS):
            for bit in range(BITS):
                model.addConstr(p[state, row, bit] <= l[state, row, bit], name=f"PL_{state}_{row}_{bit}")


def add_initial_state(model: gp.Model, p: gp.tupledict, l: gp.tupledict) -> None:
    for row in range(ROWS):
        for bit in range(BITS):
            model.addConstr(l["A0", row, bit] == 1, name=f"A0_l_{row}_{bit}")
            if row != 0 or bit == PADDING_BIT:
                model.addConstr(p["A0", row, bit] == 0, name=f"A0_p_const_{row}_{bit}")


def add_sbox_layer(
    model: gp.Model,
    p: gp.tupledict,
    l: gp.tupledict,
    value: gp.tupledict,
    src: str,
    dst: str,
    patterns: Sequence[Pattern],
    conditions: Dict[Tuple[str, int, Tuple[int, ...], int], gp.Var],
    pattern_choice: Dict[Tuple[str, int], List[gp.Var]],
) -> None:
    required_by_condition: DefaultDict[Tuple[str, int, Tuple[int, ...], int], List[gp.Var]] = defaultdict(list)

    for bit in range(BITS):
        choices = [
            model.addVar(vtype=GRB.BINARY, name=f"PAT_{src}_{dst}_x{bit}_{idx}")
            for idx in range(len(patterns))
        ]
        pattern_choice[src, bit] = choices
        model.addConstr(gp.quicksum(choices) == 1, name=f"PAT_{src}_{dst}_x{bit}_onehot")

        for row in range(ROWS):
            out_p = gp.quicksum(choices[idx] for idx, pat in enumerate(patterns) if category_pl(pat.outputs[row])[0])
            out_l = gp.quicksum(choices[idx] for idx, pat in enumerate(patterns) if category_pl(pat.outputs[row])[1])
            model.addConstr(p[dst, row, bit] == out_p, name=f"{dst}_{row}_{bit}_p_from_pat")
            model.addConstr(l[dst, row, bit] == out_l, name=f"{dst}_{row}_{bit}_l_from_pat")

        for idx, pat in enumerate(patterns):
            for condition in pat.conditions:
                key = condition_key(src, bit, condition)
                cond = add_condition(model, p, l, value, key, conditions)
                model.addConstr(
                    choices[idx] <= cond,
                    name=f"PAT_{src}_{bit}_{idx}_{condition_tag(condition)}",
                )
                required_by_condition[key].append(choices[idx])

            for row, kind in enumerate(pat.inputs):
                if kind == "q":
                    # Nonlinear means L=0.
                    model.addConstr(choices[idx] <= 1 - l[src, row, bit], name=f"{src}_{bit}_{idx}_in_q_{row}")
                elif kind == "v":
                    # Linear means P=1, and P<=L is global.
                    model.addConstr(choices[idx] <= p[src, row, bit], name=f"{src}_{bit}_{idx}_in_v_{row}")
                else:
                    # Constant means (P,L)=(0,1).
                    model.addConstr(choices[idx] <= l[src, row, bit], name=f"{src}_{bit}_{idx}_in_c_l_{row}")
                    model.addConstr(choices[idx] <= 1 - p[src, row, bit], name=f"{src}_{bit}_{idx}_in_c_p_{row}")

    for key, selectors in required_by_condition.items():
        conditions[key].setAttr("BranchPriority", 1)
        model.addConstr(conditions[key] <= gp.quicksum(selectors), name=f"COND_{key[0]}_{key[1]}_used")


def add_linear_layer(model: gp.Model, p: gp.tupledict, l: gp.tupledict, src: str, dst: str) -> None:
    for row, (r0, r1) in enumerate(ROT_OFFSETS):
        for bit in range(BITS):
            refs = [
                (row, bit),
                (row, (bit - r0) % BITS),
                (row, (bit - r1) % BITS),
            ]
            l_inputs = [l[src, y, x] for y, x in refs]
            p_inputs = [p[src, y, x] for y, x in refs]
            any_p = model.addVar(vtype=GRB.BINARY, name=f"ANY_P_{src}_{dst}_{row}_{bit}")

            add_and(model, l[dst, row, bit], l_inputs, f"L_{src}_{dst}_{row}_{bit}")
            add_or(model, any_p, p_inputs, f"ANY_P_{src}_{dst}_{row}_{bit}")
            model.addConstr(p[dst, row, bit] <= l[dst, row, bit], name=f"P_{dst}_{row}_{bit}_le_l")
            model.addConstr(p[dst, row, bit] <= any_p, name=f"P_{dst}_{row}_{bit}_le_any")
            model.addConstr(p[dst, row, bit] >= any_p + l[dst, row, bit] - 1, name=f"P_{dst}_{row}_{bit}_lb")


def add_linear_outputs(model: gp.Model, l: gp.tupledict) -> gp.tupledict:
    selected = model.addVars(BITS, vtype=GRB.BINARY, name="SEL_LINEAR_OUT")
    for bit in range(BITS):
        rows = [l["A3", row, bit] for row in (0, 2, 3)]
        for idx, item in enumerate(rows):
            model.addConstr(selected[bit] <= item, name=f"SEL_{bit}_ub_{idx}")
        model.addConstr(selected[bit] >= gp.quicksum(rows) - 2, name=f"SEL_{bit}_lb")
    return selected


def build_model(
    target_de: int | None,
    resource_bound: int | None,
    fixed_df: int | None = 6,
) -> Dict[str, object]:
    model = gp.Model("ascon_4round_linear_output")
    p = model.addVars(STATES, ROWS, BITS, vtype=GRB.BINARY, name="P")
    l = model.addVars(STATES, ROWS, BITS, vtype=GRB.BINARY, name="L")
    value = model.addVars(STATES, ROWS, BITS, vtype=GRB.BINARY, name="VAL")
    conditions: Dict[Tuple[str, int, Tuple[int, ...], int], gp.Var] = {}
    pattern_choice: Dict[Tuple[str, int], List[gp.Var]] = {}

    add_pl_partition(model, p, l)
    add_initial_state(model, p, l)

    add_sbox_layer(model, p, l, value, "A0", "B0", FIRST_PS, conditions, pattern_choice)
    add_linear_layer(model, p, l, "B0", "A1")
    add_sbox_layer(model, p, l, value, "A1", "B1", SECOND_PS_FOR_4ROUND, conditions, pattern_choice)
    add_linear_layer(model, p, l, "B1", "A2")
    add_sbox_layer(model, p, l, value, "A2", "B2", THIRD_PS_FOR_4ROUND, conditions, pattern_choice)
    add_linear_layer(model, p, l, "B2", "A3")

    selected = add_linear_outputs(model, l)
    df = gp.quicksum(p["A0", 0, bit] for bit in range(BITS))
    de = gp.quicksum(selected[bit] for bit in range(BITS))
    condition_count = gp.quicksum(conditions.values())
    a1_condition_count_excluding_y2 = gp.quicksum(
        var
        for (state, _, rows, _), var in conditions.items()
        if state == "A1" and 2 not in rows
    )
    third_round_condition_count = gp.quicksum(
        var for (state, _, _, _), var in conditions.items() if state == "A2"
    )

    model.addConstr(
        a1_condition_count_excluding_y2 <= 29,
        name="a1_condition_limit_excluding_y2",
    )
    model.addConstr(third_round_condition_count <= 9, name="third_round_condition_limit")
    model.addConstr(de <= df, name="df_ge_de")
    if fixed_df is not None:
        if not 0 <= fixed_df <= BITS - 1:
            raise ValueError(f"fixed_df must be in [0, {BITS - 1}], got {fixed_df}")
        model.addConstr(df == fixed_df, name="fixed_df")
    if target_de is not None:
        model.addConstr(de >= target_de, name="target_de")
    if resource_bound is not None:
        model.addConstr(df + condition_count <= resource_bound+5, name="resource_bound")

    # Primary objective: maximize linear-output equations.  Ties prefer fewer
    # conditions and then fewer free variables.
    objective = -de + 1e-4 * condition_count + 1e-6 * df
    model.setObjective(objective, GRB.MINIMIZE)

    return {
        "model": model,
        "P": p,
        "L": l,
        "VAL": value,
        "conditions": conditions,
        "selected": selected,
        "pattern_choice": pattern_choice,
        "df": df,
        "de": de,
        "condition_count": condition_count,
        "objective": objective,
    }


def one(var: gp.Var) -> bool:
    return var.X > 0.5


def cat(pv: gp.Var, lv: gp.Var) -> str:
    if not one(lv):
        return "q"
    if one(pv):
        return "v"
    return "c"


def start_json_path(path: Path) -> Path:
    if path.exists():
        return path
    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative
    return path


def apply_start_json(data: Dict[str, object], path: Path) -> None:
    path = start_json_path(path)
    if not path.exists():
        print(f"start_json not found, skip warm start: {path}")
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    states: Mapping[str, Sequence[str]] = raw["states"]
    p = data["P"]
    l = data["L"]
    selected = data["selected"]
    conditions = data["conditions"]
    pattern_choice = data["pattern_choice"]

    for state in STATES:
        if state not in states:
            continue
        for bit, column in enumerate(states[state]):
            for row, kind in enumerate(column):
                pv, lv = category_pl(kind)
                p[state, row, bit].Start = pv
                l[state, row, bit].Start = lv

    selected_bits = set(int(bit) for bit in raw.get("selected_linear_output_bits", []))
    for bit in range(BITS):
        selected[bit].Start = 1 if bit in selected_bits else 0

    active_conditions = {
        (
            str(item["state"]),
            int(item["bit"]),
            tuple(sorted(int(row) for row in item["rows"])),
            int(item["rhs"]),
        )
        for item in raw.get("conditions", [])
    }
    for key, var in conditions.items():
        var.Start = 1 if key in active_conditions else 0

    layers = (
        ("A0", "B0", FIRST_PS),
        ("A1", "B1", SECOND_PS_FOR_4ROUND),
        ("A2", "B2", THIRD_PS_FOR_4ROUND),
    )
    for src, dst, patterns in layers:
        if src not in states or dst not in states:
            continue
        for bit in range(BITS):
            choices = pattern_choice[src, bit]
            src_column = states[src][bit]
            dst_column = states[dst][bit]
            chosen = None
            for index, pattern in enumerate(patterns):
                if pattern.inputs != src_column or pattern.outputs != dst_column:
                    continue
                required = {
                    condition_key(src, bit, condition)
                    for condition in pattern.conditions
                }
                if required <= active_conditions:
                    chosen = index
                    break
                if chosen is None:
                    chosen = index
            if chosen is None:
                continue
            for index, choice in enumerate(choices):
                choice.Start = 1 if index == chosen else 0

    print("warm_start_json:", str(path.resolve()))


def collect_solution(data: Dict[str, object]) -> Dict[str, object]:
    p = data["P"]
    l = data["L"]
    selected = data["selected"]
    conditions = data["conditions"]

    selected_bits = [bit for bit in range(BITS) if one(selected[bit])]
    free_bits = [bit for bit in range(BITS) if one(p["A0", 0, bit])]
    active_conditions = [
        {"state": state, "bit": bit, "rows": list(rows), "rhs": rhs}
        for (state, bit, rows, rhs), var in conditions.items()
        if one(var)
    ]
    states = {
        state: [
            "".join(cat(p[state, row, bit], l[state, row, bit]) for row in range(ROWS))
            for bit in range(BITS)
        ]
        for state in STATES
    }
    de = len(selected_bits)
    return {
        "df": len(free_bits),
        "de": de,
        "condition_count": len(active_conditions),
        "complexity_exponent": 128 - ALPHA * de,
        "free_bits": free_bits,
        "selected_linear_output_bits": selected_bits,
        "conditions": active_conditions,
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="4-round Ascon MILP with P/L encoding and linear outputs.")
    parser.add_argument("--target-de", type=int, default=7, help="Minimum number of selected linear outputs.")
    parser.add_argument(
        "--fixed-df",
        type=int,
        default=7,
        help="Fix the number of free A0[0][x] variables; the paper's 4-round setting is 7.",
    )
    parser.add_argument("--resource-bound", type=int, default=64)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--write-lp", type=Path, default=None)
    parser.add_argument(
        "--start-json",
        type=Path,
        default=Path("work/ascon_4round_linear_output_7.json"),
        help="Use an existing JSON solution as a Gurobi warm start. Use an empty/nonexistent path to skip.",
    )
    parser.add_argument("--json-output", type=Path, default=Path("work/ascon_4round_linear_output_5.json"))
    args = parser.parse_args()

    data = build_model(
        target_de=args.target_de,
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

    if args.start_json:
        apply_start_json(data, args.start_json)

    model.optimize()
    if model.SolCount == 0:
        print("No feasible solution found.")
        return

    result = collect_solution(data)
    print("status:", model.Status)
    print("objective:", model.ObjVal)
    print("df:", result["df"])
    print("de:", result["de"])
    print("condition_count:", result["condition_count"])
    print("complexity_exponent:", result["complexity_exponent"])
    print("free_bits:", result["free_bits"])
    print("selected_linear_output_bits:", result["selected_linear_output_bits"])
    print("conditions:", result["conditions"])

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("json:", str(args.json_output.resolve()))


if __name__ == "__main__":
    main()
