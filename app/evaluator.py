"""Relational-algebra evaluator with minimal-witness why-provenance.

Supported plan nodes (each is an object with a ``type``):

  {"type": "scan", "table": "orders"}
  {"type": "filter", "input": <node>, "predicate": <predicate tree>}
  {"type": "project", "input": <node>,
   "columns": ["a", {"from": "b", "as": "c"}]}
  {"type": "join", "left": <node>, "right": <node>,
   "on": [{"left": "id", "right": "cid"}]}
  {"type": "aggregate", "input": <node>,
   "group_by": ["cust"],
   "aggregates": [{"fn": "sum", "column": "amount", "as": "total"}]}

Every intermediate relation is a list of *labeled rows*::

    {"data": (v1, v2, ...),
     "witnesses": set[frozenset[label]],   # minimal why-witnesses (DNF)
     "killers": set[frozenset[label]] | None}

Rows with identical data are merged (set semantics); their witness sets and
killer sets are unioned and pruned to inclusion-minimal sets. A label is a
``(table, pk-tuple)`` pair identifying one submitted source row.

Witnesses vs killers
--------------------
``witnesses`` are the standard why-provenance witnesses: a row is derivable
from a sub-snapshot containing one full witness. For the monotone operators
(scan/filter/equi-join/project) the minimal deletion sets are exactly the
minimal hitting sets of the witnesses.

Exact-value aggregates are not monotone in that sense (a surviving witness
does not protect the value from other surviving rows, e.g. ``sum`` with zeros
or mixed signs), so aggregate rows additionally carry exact ``killers``:
the inclusion-minimal sets of source rows whose deletion makes *that exact
output row* disappear on recomputation. Operators above an aggregate lift
the killers of their inputs; rows without an aggregate below them leave
``killers`` as ``None`` and fall back to hitting sets of their witnesses.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .errors import ApiError
from .values import (
    eval_predicate,
    is_number,
    strict_equal,
    tuple_sort_key,
    type_name,
)

Label = tuple[str, tuple]
Witness = frozenset[Label]

# Column descriptor: (name, declared type or None, nullable)
Col = tuple[str, str | None]


# ---------------------------------------------------------------------------
# Table loading / validation
# ---------------------------------------------------------------------------

def _coerce_value(v: Any, declared: str | None, *, where: str) -> Any:
    if v is None:
        return None
    if declared == "bool":
        if not isinstance(v, bool):
            raise ApiError("TYPE_ERROR", f"{where}: expected bool, got {type_name(v)}")
    elif declared == "int":
        if not isinstance(v, int) or isinstance(v, bool):
            raise ApiError("TYPE_ERROR", f"{where}: expected int, got {type_name(v)}")
    elif declared == "float":
        if not is_number(v):
            raise ApiError("TYPE_ERROR", f"{where}: expected number, got {type_name(v)}")
        return float(v)
    elif declared == "string":
        if not isinstance(v, str):
            raise ApiError("TYPE_ERROR", f"{where}: expected string, got {type_name(v)}")
    return v


class Table:
    def __init__(self, spec: dict):
        self.name: str = spec["table"]
        self.pk_cols: list[str] = list(spec["primary_key"])
        self.columns: list[Col] = []

        declared: dict[str, str] = {}
        for c in spec.get("columns", []):
            if isinstance(c, dict):
                declared[c["name"]] = c["type"]
                self.columns.append((c["name"], c["type"]))
            else:
                declared[c] = "any"
                self.columns.append((c, None))

        rows_in: list[dict] = spec.get("rows", [])
        # Infer columns from rows when (or beyond) declared columns.
        seen = {name for name, _ in self.columns}
        for r in rows_in:
            for k in r.keys():
                if k not in seen:
                    self.columns.append((k, None))
                    seen.add(k)

        self.colnames = [name for name, _ in self.columns]
        unknown_pk = [c for c in self.pk_cols if c not in self.colnames]
        if unknown_pk:
            raise ApiError(
                "ILLEGAL_REFERENCE",
                f"Table {self.name!r}: primary key column(s) {unknown_pk} not "
                f"among declared columns",
                table=self.name,
                columns=unknown_pk,
            )

        # rows: list of (pk-tuple, full data tuple aligned to columns, label)
        self.rows: list[tuple[tuple, tuple, Label]] = []
        seen_pks: dict[tuple, Label] = {}
        for idx, r in enumerate(rows_in):
            extra = [k for k in r if k not in self.colnames]
            if extra:
                raise ApiError(
                    "MALFORMED_REQUEST",
                    f"Table {self.name!r} row {idx}: unknown column(s) {extra}",
                    table=self.name,
                    row_index=idx,
                    columns=extra,
                )
            vals = []
            for name, typ in self.columns:
                v = r.get(name)
                vals.append(_coerce_value(v, typ, where=f"table {self.name!r} row {idx}, column {name!r}"))
            pk = tuple(r.get(c) for c in self.pk_cols)
            if any(p is None for p in pk):
                raise ApiError(
                    "NULL_PRIMARY_KEY",
                    f"Table {self.name!r} row {idx}: primary key part is null",
                    table=self.name,
                    row_index=idx,
                )
            label = (self.name, pk)
            if pk in seen_pks:
                raise ApiError(
                    "DUPLICATE_PRIMARY_KEY",
                    f"Table {self.name!r}: duplicate primary key {list(pk)}",
                    table=self.name,
                    primary_key=list(pk),
                )
            seen_pks[pk] = label
            self.rows.append((pk, tuple(vals), label))

    def filtered_copy(self, drop_pks: set[tuple]) -> "Table":
        """Same schema, rows whose primary key is not in *drop_pks*."""
        clone = Table.__new__(Table)
        clone.name = self.name
        clone.pk_cols = self.pk_cols
        clone.columns = self.columns
        clone.colnames = self.colnames
        clone.rows = [r for r in self.rows if r[0] not in drop_pks]
        return clone


# ---------------------------------------------------------------------------
# Witness utilities
# ---------------------------------------------------------------------------

def prune(family: Iterable[frozenset]) -> set[frozenset]:
    """Keep only inclusion-minimal members of a family of frozensets."""
    uniq = set(family)
    return {w for w in uniq if not any(o < w for o in uniq)}


def conjoin_families(
    left: Iterable[frozenset[int]],
    right: Iterable[frozenset[int]],
) -> set[frozenset[int]]:
    """All inclusion-minimal unions ``l | r`` (DNF conjunction operator).

    Used to combine the requirements of several aggregates in one aggregate
    node: a witness for the whole output row must satisfy every aggregate.
    """
    return prune(l | r for l in left for r in right)


def combine_breaker_clauses(
    clauses: list[set[frozenset[int]]],
) -> set[frozenset[int]]:
    """Minimal member deletions breaking at least one clause.

    The exact output row remains iff *all* requirements hold (group exists,
    count equals the target, sum equals the target, min/max equal the
    targets), so it disappears when *any* clause is broken. The minimal
    such deletions are therefore the inclusion-minimal members of the union
    of the clauses' own minimal breakers.

    Caveat: with mixed-sign sums the "sum breaks" event is not monotone
    under adding more deleted members (cancellations can restore the
    target). Callers therefore re-validate the lifted base-row sets by
    recomputation.
    """
    pooled: set[frozenset[int]] = set()
    for clause in clauses:
        pooled |= clause
    return prune(pooled)


def minimal_sum_subsets(values: list, target) -> set[frozenset[int]]:
    """All inclusion-minimal sets of indices whose ``values`` sum to *target*.

    Combinations with equal values are kept as distinct index combinations:
    rows 50(a) and 50(b) yield two different witnesses. Works for ints and
    floats (exact arithmetic, matching the evaluator's own ``sum``), negative
    values and zeros; non-minimal matches produced via zero entries are
    pruned at the end.
    """
    # Sort indices by descending magnitude so that same-sign prefix bounds are
    # tight. Index identities are preserved, so equal values stay distinct.
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))
    n = len(values)
    positive = all(values[i] > 0 for i in range(n))
    negative = all(values[i] < 0 for i in range(n))
    results: set[frozenset[int]] = set()

    def dfs(pos: int, chosen: frozenset[int], acc) -> None:
        if acc == target and chosen:
            results.add(chosen)
            # Any extension sums to target only by adding net-zero entries,
            # which can never be inclusion-minimal.
            return
        if pos >= n:
            return
        # Same-sign prefix bound: remaining magnitudes (descending) cannot
        # close a positive (resp. negative) gap past their raw sum.
        rem = order[pos:]
        if positive or negative:
            total_rem = sum(values[i] for i in rem)
            if positive:
                if target > 0 and acc > target:
                    return
                if acc + total_rem < target:
                    return
            else:
                if target < 0 and acc < target:
                    return
                if acc + total_rem > target:
                    return
        idx = rem[0]
        dfs(pos + 1, chosen | {idx}, acc + values[idx])
        dfs(pos + 1, chosen, acc)

    dfs(0, frozenset(), 0)
    return prune(results)


def merge_relation(rows: Iterable[dict]) -> list[dict]:
    """Merge labeled rows with identical data tuples; prune witnesses/killers."""
    by_data: dict[tuple, set[Witness]] = defaultdict(set)
    killers_by_data: dict[tuple, set[Witness]] = defaultdict(set)
    have_killers: dict[tuple, bool] = defaultdict(bool)
    for r in rows:
        by_data[r["data"]].update(r["witnesses"])
        if r.get("killers") is not None:
            killers_by_data[r["data"]].update(r["killers"])
            have_killers[r["data"]] = True
    out = []
    for data, ws in by_data.items():
        entry: dict = {"data": data, "witnesses": prune(ws)}
        entry["killers"] = prune(killers_by_data[data]) if have_killers[data] else None
        out.append(entry)
    out.sort(key=lambda r: tuple_sort_key(r["data"]))
    return out


# ---------------------------------------------------------------------------
# Minimal hitting sets (minimal transversals)
# ---------------------------------------------------------------------------

def _label_sort_key(label: Label):
    from .values import value_sort_key

    return (label[0], tuple(value_sort_key(p) for p in label[1]))


def minimal_hitting_sets(witnesses: Iterable[Witness]) -> list[Witness]:
    """All inclusion-minimal sets H with H ∩ W ≠ ∅ for every witness W.

    For a monotone operator these are exactly the minimal source-row
    deletion sets that make the output row disappear.
    """
    edges = tuple(frozenset(w) for w in witnesses if w)
    memo: dict[frozenset, set[Witness]] = {}

    def rec(edges: tuple[Witness, ...]) -> set[Witness]:
        key = frozenset(edges)
        if key in memo:
            return memo[key]
        if not edges:
            return {frozenset()}

        # Pick the smallest edge to branch on (keeps the recursion narrow).
        e = min(edges, key=len)
        results: set[Witness] = set()
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

    hitting = rec(edges)
    return sorted(
        hitting,
        key=lambda h: [_label_sort_key(l) for l in sorted(h, key=_label_sort_key)],
    )


def row_killers(row: dict) -> set[Witness]:
    """Exact minimal deletion sets of an output row.

    Rows produced above an aggregate node carry precomputed ``killers``;
    every other row is monotone, so its minimal deletion sets are the
    minimal hitting sets of its witnesses.
    """
    if row.get("killers") is not None:
        return row["killers"]
    return set(minimal_hitting_sets(row["witnesses"]))


class Evaluator:
    def __init__(self, tables: dict[str, Table], *, with_provenance: bool = True):
        self.tables = tables
        self.with_provenance = with_provenance
        self._cache: dict[int, list[dict]] = {}

    # -- node helpers -------------------------------------------------------

    def eval(self, node: Any) -> list[dict]:
        if not isinstance(node, dict):
            raise ApiError("MALFORMED_PLAN", "Plan node must be an object")
        nid = id(node)
        if nid in self._cache:
            return self._cache[nid]
        t = node.get("type")
        if t == "scan":
            res = self._scan(node)
        elif t == "filter":
            res = self._filter(node)
        elif t == "project":
            res = self._project(node)
        elif t == "join":
            res = self._join(node)
        elif t == "aggregate":
            res = self._aggregate(node)
        else:
            raise ApiError(
                "MALFORMED_PLAN",
                f"Unknown plan node type {t!r}; expected one of "
                "scan/filter/project/join/aggregate",
            )
        self._cache[nid] = res
        return res

    def schema(self, node: Any) -> list[str]:
        t = node.get("type")
        if t == "scan":
            table = self._require_table(node)
            return table.colnames
        if t == "filter":
            self._filter_schema(node)
            return self.schema(self._input(node))
        if t == "project":
            return [asname for _, asname in self._proj_cols(node)]
        if t == "join":
            return self._join_schema(node)[0]
        if t == "aggregate":
            return self._agg_schema(node)
        raise ApiError("MALFORMED_PLAN", f"Unknown plan node type {t!r}")

    def _input(self, node: dict) -> Any:
        if "input" not in node:
            raise ApiError("MALFORMED_PLAN", f"{node.get('type')!r} node needs 'input'")
        return node["input"]

    def _filter_schema(self, node: dict) -> None:
        """Validate predicate shape and column references against input schema."""
        pred = node.get("predicate")
        if not isinstance(pred, dict):
            raise ApiError("MALFORMED_PLAN", "filter node needs a 'predicate' object")
        names = self.schema(self._input(node))

        def check(p: Any) -> None:
            if not isinstance(p, dict) or not p:
                raise ApiError(
                    "MALFORMED_PLAN",
                    "Predicate must be a non-empty object",
                )
            compound = [k for k in p if k in ("and", "or", "not")]
            if compound:
                if len(p) != 1:
                    raise ApiError(
                        "MALFORMED_PLAN",
                        f"Compound predicate {compound[0]!r} must have no sibling keys",
                    )
                key, body = compound[0], p[compound[0]]
            elif "column" in p:
                key, body = "column", p
            else:
                raise ApiError(
                    "MALFORMED_PLAN",
                    "Predicate must have key 'and'/'or'/'not'/'column'",
                )
            if key == "and" or key == "or":
                if not isinstance(body, list) or not body:
                    raise ApiError(
                        "MALFORMED_PLAN", f"{key!r} predicate needs a non-empty list"
                    )
                for sub in body:
                    check(sub)
            elif key == "not":
                check(body)
            elif key == "column":
                if not isinstance(body, dict):
                    raise ApiError("MALFORMED_PLAN", "column predicate must be object")
                col = body.get("column")
                op = body.get("op")
                if not isinstance(col, str) or not isinstance(op, str):
                    raise ApiError(
                        "MALFORMED_PLAN", "Predicate needs string 'column' and 'op'"
                    )
                if col not in names:
                    raise ApiError(
                        "ILLEGAL_REFERENCE",
                        f"Predicate references unknown column {col!r}",
                        column=col,
                        available=names,
                    )
                if op in {"is_null", "is_not_null"}:
                    return
                if op not in {"=", "<>", "<", "<=", ">", ">="}:
                    raise ApiError("MALFORMED_PLAN", f"Unknown predicate operator {op!r}")
                if "value" not in body:
                    raise ApiError("MALFORMED_PLAN", f"Predicate {op!r} needs 'value'")
                lit = body["value"]
                if not isinstance(lit, (str, int, float, bool)) and lit is not None:
                    raise ApiError(
                        "MALFORMED_PLAN",
                        "Predicate literal must be a JSON scalar or null",
                    )
            else:
                raise ApiError("MALFORMED_PLAN", f"Unknown predicate node {key!r}")

        check(pred)

    def _require_table(self, node: dict) -> Table:
        name = node.get("table")
        if not isinstance(name, str):
            raise ApiError("MALFORMED_PLAN", "scan node needs string 'table'")
        table = self.tables.get(name)
        if table is None:
            raise ApiError(
                "UNKNOWN_TABLE",
                f"Plan references table {name!r} which was not submitted",
                table=name,
            )
        return table

    def _col_index(self, node: Any, col: str) -> tuple[int, list[str]]:
        names = self.schema(node)
        if col not in names:
            raise ApiError(
                "ILLEGAL_REFERENCE",
                f"Column {col!r} not found in node output columns {names}",
                column=col,
                available=names,
            )
        return names.index(col), names

    # -- operators ----------------------------------------------------------

    def _scan(self, node: dict) -> list[dict]:
        table = self._require_table(node)
        return [
            {"data": data, "witnesses": {frozenset({label})}}
            for _, data, label in table.rows
        ]

    def _filter(self, node: dict) -> list[dict]:
        src = self.eval(self._input(node))
        pred = node.get("predicate")
        if not isinstance(pred, dict):
            raise ApiError("MALFORMED_PLAN", "filter node needs a 'predicate' object")
        names = self.schema(self._input(node))
        out = []
        for r in src:
            row = dict(zip(names, r["data"]))
            keep = eval_predicate(pred, row)
            if keep is True:
                out.append(r)
        # filter cannot introduce witness redundancy, but keep ordering stable
        out.sort(key=lambda r: tuple_sort_key(r["data"]))
        return out

    def _proj_cols(self, node: dict) -> list:
        cols = node.get("columns")
        if not isinstance(cols, list) or not cols:
            raise ApiError(
                "MALFORMED_PLAN", "project node needs a non-empty 'columns' list"
            )
        norm = []
        for c in cols:
            if isinstance(c, str):
                norm.append((c, c))
            elif isinstance(c, dict) and isinstance(c.get("from"), str) and isinstance(c.get("as"), str):
                norm.append((c["from"], c["as"]))
            else:
                raise ApiError(
                    "MALFORMED_PLAN",
                    "project columns must be strings or {from, as} objects",
                )
        return norm

    def _proj_as(self, c: Any) -> str:
        return c if isinstance(c, str) else c["as"]

    def _project(self, node: dict) -> list[dict]:
        src_node = self._input(node)
        src = self.eval(src_node)
        names = self.schema(src_node)
        spec = self._proj_cols(node)
        idx: list[int] = []
        out_names: list[str] = []
        for frm, asname in spec:
            if frm not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"project references unknown column {frm!r}",
                    column=frm,
                    available=names,
                )
            if asname in out_names:
                raise ApiError(
                    "MALFORMED_PLAN", f"project output has duplicate column {asname!r}"
                )
            idx.append(names.index(frm))
            out_names.append(asname)
        produced = []
        for r in src:
            produced.append(
                {
                    "data": tuple(r["data"][i] for i in idx),
                    "witnesses": set(r["witnesses"]),
                    "killers": set(r["killers"]) if r.get("killers") is not None else None,
                }
            )
        return merge_relation(produced)

    def _join_aliases(self, node: dict) -> tuple[str | None, str | None]:
        left_alias = node.get("left_alias")
        right_alias = node.get("right_alias")
        for a in (left_alias, right_alias):
            if a is not None and not isinstance(a, str):
                raise ApiError("MALFORMED_PLAN", "join alias must be a string")
        return left_alias, right_alias

    def _join_schema(self, node: dict) -> tuple[list[str], list[str], list[str], list[tuple[str, str]]]:
        left_node = node.get("left")
        right_node = node.get("right")
        if left_node is None or right_node is None:
            raise ApiError("MALFORMED_PLAN", "join node needs 'left' and 'right'")
        ln = self.schema(left_node)
        rn = self.schema(right_node)
        la, ra = self._join_aliases(node)
        on = self._join_on(node, ln, rn)

        out: list[str] = list(ln)
        # Equal columns on the join keys appear once (USING semantics).
        right_key_cols = {rc for _, rc in on}
        for c in rn:
            if c in right_key_cols and c in ln:
                continue
            name = c
            if name in ln or (la and name.startswith(f"{la}.")) or (ra and name.startswith(f"{ra}.")):
                q = ra or "right"
                name = f"{q}.{c}"
                if name in ln or name in out:
                    raise ApiError(
                        "MALFORMED_PLAN",
                        f"join output column name collision on {name!r}; use "
                        "right_alias or project to disambiguate",
                    )
            out.append(name)
        return out, ln, rn, on

    def _join_on(self, node: dict, ln: list[str], rn: list[str]) -> list[tuple[str, str]]:
        on = node.get("on")
        if not isinstance(on, list) or not on:
            raise ApiError("MALFORMED_PLAN", "join node needs a non-empty 'on' list")
        pairs: list[tuple[str, str]] = []
        for p in on:
            if not isinstance(p, dict) or "left" not in p or "right" not in p:
                raise ApiError(
                    "MALFORMED_PLAN", "join 'on' entries must be {left, right} objects"
                )
            lc, rc = p["left"], p["right"]
            if lc not in ln:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"join references unknown left column {lc!r}",
                    column=lc,
                    available=ln,
                )
            if rc not in rn:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"join references unknown right column {rc!r}",
                    column=rc,
                    available=rn,
                )
            pairs.append((lc, rc))
        return pairs

    def _join(self, node: dict) -> list[dict]:
        out_names, ln, rn, on = self._join_schema(node)
        left = self.eval(node["left"])
        right = self.eval(node["right"])
        li = [ln.index(lc) for lc, _ in on]
        ri = [rn.index(rc) for _, rc in on]

        # Hash right side on the composite equi-key. Keys containing NULL never
        # match (SQL NULL semantics); rows of different hash values simply fail
        # to join. A TYPE_ERROR is only raised for colliding keys whose values
        # are equal by Python hash/eq but not strict JSON equality (e.g. bool
        # vs int), since that is an ambiguous, almost certainly unintended
        # cross-type comparison.
        index: dict[tuple, list[dict]] = defaultdict(list)
        for r in right:
            key = tuple(r["data"][i] for i in ri)
            if any(k is None for k in key):
                continue
            index[key].append(r)

        produced = []
        for lr in left:
            lkey = tuple(lr["data"][i] for i in li)
            if any(k is None for k in lkey):
                continue
            for rr in index.get(lkey, ()):
                # strict type check across the key pairs
                for a, b in zip(lkey, (rr["data"][i] for i in ri)):
                    if strict_equal(a, b) is False:
                        raise ApiError(
                            "TYPE_ERROR",
                            "Equi-join key values have mismatched types "
                            f"({type_name(a)} vs {type_name(b)})",
                        )
                data = list(lr["data"])
                right_key_cols = {rc for _, rc in on}
                for j, c in enumerate(rn):
                    if c in right_key_cols and c in ln:
                        continue
                    data.append(rr["data"][j])
                witnesses: set[Witness] = set()
                for lw in lr["witnesses"]:
                    for rw in rr["witnesses"]:
                        witnesses.add(frozenset(lw | rw))
                # A joined tuple disappears when its left contribution or
                # its right contribution is killed; the two families are
                # alternatives (disjunction, pruned). A side without an
                # aggregate below it has no precomputed killers, in which
                # case the whole joined row falls back to witness hitting
                # sets.
                lk = lr.get("killers")
                rk = rr.get("killers")
                if lk is not None or rk is not None:
                    killers: set[Witness] | None = set()
                    if lk is not None:
                        killers.update(lk)
                    if rk is not None:
                        killers.update(rk)
                    killers = prune(killers)
                else:
                    killers = None
                produced.append({"data": tuple(data), "witnesses": witnesses, "killers": killers})
        return merge_relation(produced)

    # -- aggregate ----------------------------------------------------------

    _AGG_FNS = {"count", "sum", "min", "max"}

    def _agg_spec(self, node: dict):
        aggs = node.get("aggregates", [])
        if not isinstance(aggs, list) or not aggs:
            raise ApiError(
                "MALFORMED_PLAN", "aggregate node needs a non-empty 'aggregates' list"
            )
        norm = []
        for a in aggs:
            if not isinstance(a, dict) or not isinstance(a.get("fn"), str):
                raise ApiError("MALFORMED_PLAN", "aggregate entries need string 'fn'")
            fn = a["fn"]
            if fn not in self._AGG_FNS:
                raise ApiError(
                    "MALFORMED_PLAN",
                    f"Unsupported aggregate {fn!r}; allowed: count/sum/min/max",
                )
            col = a.get("column")
            if fn != "count" and not isinstance(col, str):
                raise ApiError(
                    "MALFORMED_PLAN", f"{fn} aggregate requires a string 'column'"
                )
            if fn == "count" and col is not None and not isinstance(col, str):
                raise ApiError(
                    "MALFORMED_PLAN", "count column must be a string or '*'"
                )
            asname = a.get("as")
            if not isinstance(asname, str) or not asname:
                raise ApiError("MALFORMED_PLAN", "aggregate requires string 'as'")
            norm.append((fn, col, asname))
        names = [a[2] for a in norm]
        if len(set(names)) != len(names):
            raise ApiError("MALFORMED_PLAN", "aggregate output aliases must be unique")
        return norm

    def _agg_schema(self, node: dict) -> list[str]:
        gb = node.get("group_by", [])
        if not isinstance(gb, list) or not all(isinstance(c, str) for c in gb):
            raise ApiError("MALFORMED_PLAN", "aggregate 'group_by' must be strings")
        out = list(gb) + [a[2] for a in self._agg_spec(node)]
        if len(set(out)) != len(out):
            raise ApiError(
                "MALFORMED_PLAN", "aggregate output column names must be unique"
            )
        return out

    def _aggregate(self, node: dict) -> list[dict]:
        src_node = self._input(node)
        gb = node.get("group_by", [])
        aggs = self._agg_spec(node)
        names = self.schema(src_node)

        for c in gb:
            if c not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"group_by references unknown column {c!r}",
                    column=c,
                    available=names,
                )
        for fn, col, asname in aggs:
            if fn != "count" and col not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"aggregate {asname!r} references unknown column {col!r}",
                    column=col,
                    available=names,
                )
            if fn == "count" and col not in (None, "*") and col not in names:
                raise ApiError(
                    "ILLEGAL_REFERENCE",
                    f"aggregate {asname!r} references unknown column {col!r}",
                    column=col,
                    available=names,
                )

        src = self.eval(src_node)

        if not src:
            # Aggregating an empty relation is always rejected: for grouped
            # aggregates it would silently yield zero groups, for scalar
            # aggregates a single row of NULL/0 — the client must distinguish
            # "no input" explicitly.
            raise ApiError(
                "EMPTY_AGGREGATE",
                "aggregate input is empty: no groups can be formed",
            )

        gi = [names.index(c) for c in gb]

        # NULL group keys form their own group. We key the dict with a
        # sentinel-prefixed tuple so a leading literal value can never collide
        # with the marker, and restore the real key when emitting.
        NULL_KEY = object()
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in src:
            gkey = tuple(r["data"][i] for i in gi)
            if any(k is None for k in gkey):
                groups[(NULL_KEY,) + gkey].append(r)
            else:
                groups[gkey].append(r)

        out_rows = []
        for gkey, members in groups.items():
            real_key = gkey[1:] if gkey and gkey[0] is NULL_KEY else gkey
            data: list[Any] = list(real_key)

            # Value-only mode (used while validating provenance by
            # recomputation): compute aggregate values without any
            # witness/killer bookkeeping.
            if not self.with_provenance:
                self._append_aggregate_values(data, members, names, aggs)
                out_rows.append(
                    {"data": tuple(data), "witnesses": set(), "killers": None}
                )
                continue

            # Member-level provenance, as index sets into ``members``:
            #   good_family  — inclusion-minimal member subsets that derive
            #                  *this exact output row* (one DNF conjunction
            #                  per member subset; group existence is required);
            #   bad_family   — inclusion-minimal member subsets whose deletion
            #                  makes *this exact output row* disappear.
            # Each aggregate contributes one requirement to each family.
            good_family: set[frozenset[int]] = {frozenset()}
            # One clause per aggregate: the alternatives that change that
            # aggregate's value. Group-emptiness is appended as a final clause.
            bad_clauses: list[set[frozenset[int]]] = []

            def col_values(column: str) -> list[Any]:
                ci = names.index(column)
                return [m["data"][ci] for m in members]

            for fn, col, asname in aggs:
                if fn == "count":
                    if col is None or col == "*":
                        eligible = list(range(len(members)))
                    else:
                        vals_col = col_values(col)
                        eligible = [i for i, v in enumerate(vals_col) if v is not None]
                    target = len(eligible)
                    data.append(target)
                    # count follows the usual count(*)/count(column) rule:
                    # every counted row participates in the single witness;
                    # killing any one of them changes the count.
                    req_good = {frozenset(eligible)} if eligible else {frozenset()}
                    req_bad = {frozenset({i}) for i in eligible}
                    good_family = conjoin_families(good_family, req_good)
                    if req_bad:
                        bad_clauses.append(req_bad)
                    continue

                raw = col_values(col)
                vals = [(i, v) for i, v in enumerate(raw) if v is not None]
                if fn == "sum":
                    bad_v = [v for _, v in vals if not is_number(v)]
                    if bad_v:
                        raise ApiError(
                            "TYPE_ERROR",
                            f"sum({col!r}): non-numeric value "
                            f"{bad_v[0]!r} ({type_name(bad_v[0])})",
                            column=col,
                        )
                    target = sum(v for _, v in vals) if vals else None
                    data.append(target)
                    if vals:
                        idx_order = [i for i, _ in vals]
                        nums = [v for _, v in vals]
                        combos = minimal_sum_subsets(nums, target)
                        req_good = {
                            frozenset(idx_order[k] for k in combo) for combo in combos
                        }
                        # sum changes iff deleted non-null values have a
                        # nonzero sum; the minimal such subsets are singletons
                        # over nonzero values (a zero-only subset sums to 0
                        # and never matters).
                        req_bad = {
                            frozenset({idx_order[j]})
                            for j, v in enumerate(nums)
                            if v != 0
                        }
                    else:
                        # NULL sum: any single all-NULL member witnesses it;
                        # a NULL must remain (the group may stay non-empty).
                        req_good = {frozenset({i}) for i, _ in vals} or {frozenset()}
                        req_bad = set()
                    good_family = conjoin_families(good_family, req_good)
                    if req_bad:
                        bad_clauses.append(req_bad)
                elif fn in ("min", "max"):
                    kinds = {type_name(v) for _, v in vals}
                    non_numeric = kinds - {"number"}
                    if non_numeric and ({"number"} & kinds or len(non_numeric) > 1):
                        raise ApiError(
                            "TYPE_ERROR",
                            f"{fn}({col!r}): cannot order mixed types "
                            f"{sorted(kinds)}",
                            column=col,
                        )
                    target = (
                        (min if fn == "min" else max)(v for _, v in vals)
                        if vals
                        else None
                    )
                    data.append(target)
                    if vals:
                        extremal = {i for i, v in vals if v == target}
                        # Any one extremal member produces the target
                        # extremum; the target disappears only once *all*
                        # extremal members are deleted.
                        req_good = {frozenset({i}) for i in extremal}
                        req_bad = {frozenset(extremal)}
                    else:
                        req_good = {frozenset({i}) for i, _ in vals} or {frozenset()}
                        req_bad = set()
                    good_family = conjoin_families(good_family, req_good)
                    if req_bad:
                        bad_clauses.append(req_bad)

            # Group existence: at least one member must survive. Conjoin the
            # group-existence requirement (the member singletons) with the
            # aggregate requirements, dropping non-minimal combinations.
            existence = {frozenset({i}) for i in range(len(members))}
            good_family = conjoin_families(good_family, existence)
            # The exact output row disappears as soon as *one* requirement
            # breaks (count changes / sum changes / an extremum disappears /
            # the group empties): pool the clauses' minimal breakers.
            bad_clauses.append({frozenset(range(len(members)))})
            bad_family = combine_breaker_clauses(bad_clauses)

            target_data = tuple(data)
            row_witnesses, row_killers = self._lift_aggregate_provenance(
                node, members, good_family, bad_family, target_data
            )

            out_rows.append(
                {
                    "data": target_data,
                    "witnesses": row_witnesses,
                    "killers": row_killers,
                }
            )
        return merge_relation(out_rows)

    # -- aggregate provenance lifting --------------------------------------

    @staticmethod
    def _append_aggregate_values(
        data: list[Any],
        members: list[dict],
        names: list[str],
        aggs: list,
    ) -> None:
        """Append count/sum/min/max values for one group onto *data*."""

        def col_values(column: str) -> list[Any]:
            ci = names.index(column)
            return [m["data"][ci] for m in members]

        for fn, col, _asname in aggs:
            if fn == "count":
                if col is None or col == "*":
                    eligible = members
                else:
                    eligible = [v for v in col_values(col) if v is not None]
                data.append(len(eligible))
                continue
            vals = [v for v in col_values(col) if v is not None]
            if fn == "sum":
                bad = [v for v in vals if not is_number(v)]
                if bad:
                    raise ApiError(
                        "TYPE_ERROR",
                        f"sum({col!r}): non-numeric value "
                        f"{bad[0]!r} ({type_name(bad[0])})",
                        column=col,
                    )
                data.append(sum(vals) if vals else None)
            else:  # min / max
                kinds = {type_name(v) for v in vals}
                non_numeric = kinds - {"number"}
                if non_numeric and ({"number"} & kinds or len(non_numeric) > 1):
                    raise ApiError(
                        "TYPE_ERROR",
                        f"{fn}({col!r}): cannot order mixed types "
                        f"{sorted(kinds)}",
                        column=col,
                    )
                data.append((min if fn == "min" else max)(vals) if vals else None)

    # Max size of a group's source-label universe for which every sub-snapshot
    # is enumerated exactly. 2**12 = 4096 recomputations per group.
    _EXACT_SWEEP_LIMIT = 12

    def _lift_aggregate_provenance(
        self,
        node: dict,
        members: list[dict],
        good_family: set[frozenset[int]],
        bad_family: set[frozenset[int]],
        target_data: tuple,
    ) -> tuple[set[Witness], set[Witness]]:
        """Lift member-level witness/killer families to source-label sets.

        Simple groups (every member has one singleton witness, labels
        disjoint — e.g. groups directly over a scan/filter) lift
        combinatorially and the result is exact.

        Complex groups (members derived through joins, self-joins or
        sub-aggregates, where witnesses may share base rows) are validated
        by actually recomputing the plan on the reduced snapshot, because
        shared labels and non-monotone aggregate values (mixed-sign sums)
        can invalidate a naive lift. For small label universes an exhaustive
        sub-snapshot sweep gives the exact answer; otherwise only validated,
        inclusion-minimal lifted candidates are reported.
        """
        lifted_w: set[Witness] = set()
        for combo in good_family:
            lifted_w.update(self._lift_good(combo, members))
        lifted_k: set[Witness] = set()
        for deleted in bad_family:
            lifted_k.update(self._lift_bad(deleted, members))

        universe: set[Label] = set()
        for m in members:
            for w in m["witnesses"]:
                universe.update(w)

        simple = len(members) == len(universe) and all(
            len(m["witnesses"]) == 1 and len(next(iter(m["witnesses"]))) == 1
            for m in members
        )
        if simple:
            return prune(lifted_w), prune(lifted_k)

        if len(universe) <= self._EXACT_SWEEP_LIMIT:
            return self._exact_provenance_sweep(node, universe, target_data)

        validated_w = {
            w for w in prune(lifted_w)
            if self._target_present(node, universe - w, target_data)
        }
        validated_k = {
            h for h in prune(lifted_k)
            if not self._target_present(node, h, target_data)
        }
        return prune(validated_w), prune(validated_k)

    def _exact_provenance_sweep(
        self,
        node: dict,
        universe: set[Label],
        target_data: tuple,
    ) -> tuple[set[Witness], set[Witness]]:
        """Enumerate every subset of *universe* and read provenance off it.

        A surviving subset under which the target is present is a witness
        candidate; a deleted subset under which it is absent is a killer
        candidate. Both families are pruned to inclusion-minimal members,
        so the result is exact (and automatically consistent with
        recomputation), including NULLs, ties and mixed-sign sums.
        """
        labels = list(universe)
        n = len(labels)
        witnesses: set[Witness] = set()
        killers: set[Witness] = set()
        for mask in range(1 << n):
            deleted = frozenset(labels[i] for i in range(n) if not (mask >> i) & 1)
            if self._target_present(node, deleted, target_data):
                witnesses.add(frozenset(universe - deleted))
            else:
                killers.add(deleted)
        return prune(witnesses), prune(killers)

    def _target_present(
        self, node: dict, deleted: frozenset[Label], target_data: tuple
    ) -> bool:
        """Recompute the full aggregate node after deleting *deleted*."""
        by_table: dict[str, set[tuple]] = {}
        for lab in deleted:
            by_table.setdefault(lab[0], set()).add(lab[1])
        filtered = {
            name: t.filtered_copy(by_table.get(name, set()))
            for name, t in self.tables.items()
        }
        try:
            out = Evaluator(filtered, with_provenance=False).eval(node)
        except ApiError:
            # Empty aggregate input (group cannot be formed at all).
            return False
        return any(r["data"] == target_data for r in out)

    @staticmethod
    def _lift_good(combo: frozenset[int], members: list[dict]) -> set[Witness]:
        """Witness conjunctions of one why-witness per member in *combo*."""
        lifted: set[Witness] = {frozenset()}
        for i in combo:
            lifted = {
                frozenset(w | v)
                for w in lifted
                for v in members[i]["witnesses"]
            }
        return prune(lifted)

    @staticmethod
    def _lift_bad(deleted: frozenset[int], members: list[dict]) -> set[Witness]:
        """Source-row deletions that eliminate every member in *deleted*.

        Each member must lose all of its derivations, so its witness
        hypergraph must be hit; choices across members combine conjunctively.
        """
        lifted: set[Witness] = {frozenset()}
        for i in deleted:
            hits = set(minimal_hitting_sets(members[i]["witnesses"]))
            lifted = {
                frozenset(h | v)
                for h in hits
                for v in lifted
            }
            lifted = prune(lifted)
        return lifted
