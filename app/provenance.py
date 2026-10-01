"""Provenance presentation and counterfactual (minimal deletion sets).

Boolean provenance
------------------
Each submitted source row receives a short symbol ``s1, s2, ...``. A minimal
witness set is a conjunction of symbols; the provenance of an output row is
the DNF over all its minimal witnesses::

    (s1 ∧ s3) ∨ (s2 ∧ s3)

Because witnesses are inclusion-minimal and the query is monotone
(select / equi-join / project / count-sum-min-max over groups), the sets of
source rows whose removal makes an output row disappear are exactly the
minimal hitting sets of its minimal witnesses.
"""
from __future__ import annotations

from typing import Any

from .evaluator import Label, Witness
from .values import tuple_sort_key, value_sort_key


# ---------------------------------------------------------------------------
# Labels / symbols
# ---------------------------------------------------------------------------

def label_to_string(label: Label) -> str:
    table, pk = label
    parts = [_pk_part(p) for p in pk]
    return f"{table}[" + ",".join(parts) + "]"


def _pk_part(p: Any) -> str:
    if isinstance(p, str):
        return p
    if p is None:
        return "null"
    if isinstance(p, bool):
        return "true" if p else "false"
    return str(p)


def label_sort_key(label: Label):
    return (label[0], tuple(value_sort_key(p) for p in label[1]))


def build_symbol_table(all_labels: set[Label]) -> dict[Label, str]:
    ordered = sorted(all_labels, key=label_sort_key)
    return {label: f"s{i + 1}" for i, label in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Row presentation
# ---------------------------------------------------------------------------

def present_row(
    row: dict,
    columns: list[str],
    symbols: dict[Label, str],
) -> dict:
    witnesses = sorted(
        sorted(w, key=lambda lab: label_sort_key(lab)) for w in row["witnesses"]
    )
    witness_objs = [
        {
            "labels": [label_to_string(lab) for lab in w],
            "symbols": [symbols[lab] for lab in w],
            "conjunction": " ∧ ".join(symbols[lab] for lab in w),
        }
        for w in witnesses
    ]
    return {
        "row": {c: v for c, v in zip(columns, row["data"])},
        "witness_count": len(witness_objs),
        "minimal_witnesses": witness_objs,
        "provenance": {
            "dnf": " ∨ ".join(f"({w['conjunction']})" for w in witness_objs),
            "dnf_structured": [w["symbols"] for w in witness_objs],
            "labels_structured": [w["labels"] for w in witness_objs],
        },
    }


# ---------------------------------------------------------------------------
# Minimal hitting sets (minimal transversals)
# ---------------------------------------------------------------------------

def minimal_hitting_sets(witnesses: set[Witness]) -> list[frozenset[Label]]:
    """All inclusion-minimal sets H with H ∩ W ≠ ∅ for every W."""
    edges = [frozenset(w) for w in witnesses if w]
    memo: dict[frozenset, set[frozenset[Label]]] = {}

    def rec(edges: tuple[frozenset, ...]) -> set[frozenset[Label]]:
        key = frozenset(edges)
        if key in memo:
            return memo[key]
        if not edges:
            return {frozenset()}

        # Pick the smallest edge to branch on (keeps the recursion narrow).
        e = min(edges, key=len)
        results: set[frozenset[Label]] = set()
        for v in e:
            # Choosing v covers e and every edge containing v; the residual
            # hypergraph is all remaining edges (intact) that v does not hit.
            remaining = tuple(x for x in edges if v not in x)
            for sub in rec(remaining):
                results.add(frozenset(sub | {v}))
        # Prune non-minimal solutions introduced on this branch.
        minimal = {h for h in results if not any(o < h for o in results)}
        memo[key] = minimal
        return minimal

    hitting = rec(tuple(edges))

    def hit_key(h: frozenset[Label]):
        return [label_sort_key(l) for l in sorted(h, key=label_sort_key)]

    return sorted(hitting, key=hit_key)
