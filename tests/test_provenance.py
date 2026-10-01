"""Unit tests for minimal hitting sets and witness construction."""
from app.evaluator import (
    Evaluator,
    Table,
    minimal_hitting_sets,
    minimal_sum_subsets,
    row_killers,
)


def L(table, *pk):
    return (table, pk)


def test_hitting_sets_basic():
    witnesses = {frozenset([L("t", 1), L("t", 2)]), frozenset([L("t", 2), L("t", 3)])}
    hs = minimal_hitting_sets(witnesses)
    assert frozenset([L("t", 2)]) in hs
    assert frozenset([L("t", 1), L("t", 3)]) in hs
    assert len(hs) == 2


def test_hitting_sets_nested():
    witnesses = {
        frozenset([L("t", 1), L("t", 2)]),
        frozenset([L("t", 2), L("t", 3)]),
        frozenset([L("t", 3), L("t", 4)]),
    }
    hs = minimal_hitting_sets(witnesses)
    # {2,3}, {1,3}, {2,4}
    assert len(hs) == 3
    for h in hs:
        for w in witnesses:
            assert h & w
        assert not any(o < h for o in hs)


def test_single_witness():
    w = frozenset([L("a", "x"), L("b", "y")])
    hs = minimal_hitting_sets(witnesses={w})
    assert set(hs) == {frozenset([L("a", "x")]), frozenset([L("b", "y")])}


def make_evaluator(spec):
    return Evaluator({t["table"]: Table(t) for t in spec})


def join_query():
    return {
        "type": "join",
        "left": {"type": "scan", "table": "orders"},
        "right": {"type": "scan", "table": "customers"},
        "on": [{"left": "cust", "right": "id"}],
    }


def test_join_provenance():
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "cust", "amt"],
            "rows": [
                {"oid": 1, "cust": 10, "amt": 5},
                {"oid": 2, "cust": 10, "amt": 7},
                {"oid": 3, "cust": 20, "amt": 9},
            ],
        },
        {
            "table": "customers",
            "primary_key": ["id"],
            "columns": ["id", "name"],
            "rows": [{"id": 10, "name": "A"}, {"id": 20, "name": "B"}],
        },
    ]
    ev = make_evaluator(tables)
    rows = ev.eval(join_query())
    # find joined row oid=1
    cols = ev.schema(join_query())
    r1 = next(r for r in rows if r["data"][cols.index("oid")] == 1)
    assert r1["witnesses"] == {frozenset([L("orders", 1), L("customers", 10)])}
    assert len(rows) == 3


def test_aggregate_group_witnesses():
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "cust", "amt"],
            "rows": [
                {"oid": 1, "cust": 10, "amt": 5},
                {"oid": 2, "cust": 10, "amt": 7},
                {"oid": 3, "cust": 20, "amt": 9},
            ],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "group_by": ["cust"],
        "aggregates": [
            {"fn": "count", "column": "*", "as": "n"},
            {"fn": "sum", "column": "amt", "as": "total"},
            {"fn": "min", "column": "amt", "as": "lo"},
            {"fn": "max", "column": "amt", "as": "hi"},
        ],
    }
    ev = make_evaluator(tables)
    rows = ev.eval(plan)
    cols = ev.schema(plan)
    g10 = next(r for r in rows if r["data"][cols.index("cust")] == 10)
    # count(*) requires both counted rows; sum=12 requires both 5 and 7.
    # The single minimal witness is the conjunction of all members.
    assert g10["witnesses"] == {frozenset([L("orders", 1), L("orders", 2)])}
    data = dict(zip(cols, g10["data"]))
    assert data == {"cust": 10, "n": 2, "total": 12, "lo": 5, "hi": 7}

    # Deleting either member changes count/sum (and min/max tie-break
    # requirements contribute the all-ties killer); the exact minimal
    # deletion sets are the two singletons, which agrees with recomputation.
    assert set(row_killers(g10)) == {
        frozenset([L("orders", 1)]),
        frozenset([L("orders", 2)]),
    }


def test_sum_only_witnesses_and_killers():
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "amt"],
            "rows": [{"oid": 1, "amt": 50}, {"oid": 2, "amt": 30}],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "aggregates": [{"fn": "sum", "column": "amt", "as": "total"}],
    }
    ev = make_evaluator(tables)
    r = ev.eval(plan)[0]
    assert r["data"] == (80,)
    # 50+30: only the conjunction of both rows minimally sums to 80.
    assert r["witnesses"] == {frozenset([L("orders", 1), L("orders", 2)])}
    # Either deletion alone changes the sum.
    assert row_killers(r) == {
        frozenset([L("orders", 1)]),
        frozenset([L("orders", 2)]),
    }


def test_sum_equal_value_combos_kept():
    # 30+30+50 = 110 overall; witnessing the value 80 via a sub-group has
    # two distinct index combinations (which 30 row participates matters).
    assert minimal_sum_subsets([30, 30, 50], 80) == {
        frozenset({0, 2}),
        frozenset({1, 2}),
    }
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "amt"],
            "rows": [
                {"oid": 1, "amt": 30},
                {"oid": 2, "amt": 30},
                {"oid": 3, "amt": 50},
            ],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "aggregates": [{"fn": "sum", "column": "amt", "as": "total"}],
    }
    ev = make_evaluator(tables)
    r = ev.eval(plan)[0]
    assert r["data"] == (110,)
    # Full-group target 110 needs all three rows; zero rows never form
    # spurious witnesses.
    assert r["witnesses"] == {
        frozenset([L("orders", 1), L("orders", 2), L("orders", 3)])
    }
    assert row_killers(r) == {
        frozenset([L("orders", 1)]),
        frozenset([L("orders", 2)]),
        frozenset([L("orders", 3)]),
    }


def test_min_max_tie_witnesses_and_killers():
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "amt"],
            "rows": [
                {"oid": 1, "amt": 30},
                {"oid": 2, "amt": 30},
                {"oid": 3, "amt": 50},
            ],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "aggregates": [
            {"fn": "min", "column": "amt", "as": "lo"},
            {"fn": "max", "column": "amt", "as": "hi"},
        ],
    }
    r = make_evaluator(tables).eval(plan)[0]
    assert r["data"] == (30, 50)
    # min=30 via either tie; max=50 needs row 3: two minimal witnesses.
    assert r["witnesses"] == {
        frozenset([L("orders", 1), L("orders", 3)]),
        frozenset([L("orders", 2), L("orders", 3)]),
    }
    # Kill the max row, or delete BOTH min ties.
    assert row_killers(r) == {
        frozenset([L("orders", 3)]),
        frozenset([L("orders", 1), L("orders", 2)]),
    }


def test_count_column_ignores_null():
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "amt"],
            "rows": [
                {"oid": 1, "amt": 50},
                {"oid": 2, "amt": None},
                {"oid": 3, "amt": 30},
            ],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "aggregates": [
            {"fn": "count", "column": "amt", "as": "n"},
            {"fn": "sum", "column": "amt", "as": "total"},
        ],
    }
    r = make_evaluator(tables).eval(plan)[0]
    assert r["data"] == (2, 80)
    # The NULL row neither participates in count(column) nor in sum.
    assert r["witnesses"] == {frozenset([L("orders", 1), L("orders", 3)])}
    assert row_killers(r) == {
        frozenset([L("orders", 1)]),
        frozenset([L("orders", 3)]),
    }


def test_zero_value_sum_does_not_pollute():
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "amt"],
            "rows": [{"oid": 1, "amt": 0}, {"oid": 2, "amt": 80}],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "aggregates": [{"fn": "sum", "column": "amt", "as": "total"}],
    }
    r = make_evaluator(tables).eval(plan)[0]
    assert r["data"] == (80,)
    # The zero row is not part of a minimal witness and deleting it does
    # not change the sum.
    assert r["witnesses"] == {frozenset([L("orders", 2)])}
    assert row_killers(r) == {frozenset([L("orders", 2)])}


def test_mixed_sign_sum_killers_agree_with_recompute():
    # 50 + 30 + (-30) = 50: deleting the +30 row changes the sum (to 20)
    # even though it belongs to no minimal sum-50 witness. Exact killers
    # capture this where a witness hitting set would not.
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "amt"],
            "rows": [
                {"oid": 1, "amt": 50},
                {"oid": 2, "amt": 30},
                {"oid": 3, "amt": -30},
            ],
        }
    ]
    spec = tables[0]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "aggregates": [{"fn": "sum", "column": "amt", "as": "total"}],
    }
    ev = make_evaluator(tables)
    r = ev.eval(plan)[0]
    assert r["data"] == (50,)
    assert r["witnesses"] == {frozenset([L("orders", 1)])}
    killers = row_killers(r)
    assert killers == {
        frozenset([L("orders", 1)]),
        frozenset([L("orders", 2)]),
        frozenset([L("orders", 3)]),
    }
    # Every claimed killer really removes the exact target row.
    for h in killers:
        drop = {lab[1] for lab in h}
        filtered = {"orders": Table({**spec, "rows": [
            row for row in spec["rows"] if (row["oid"],) not in drop
        ]})}
        remain = Evaluator(filtered).eval(plan)
        assert all(rr["data"] != (50,) for rr in remain)


def test_multiple_groups_independent():
    tables = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "cid", "amt"],
            "rows": [
                {"oid": 1, "cid": 10, "amt": 50},
                {"oid": 2, "cid": 10, "amt": 30},
                {"oid": 3, "cid": 20, "amt": 7},
                {"oid": 4, "cid": 20, "amt": 9},
            ],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {"type": "scan", "table": "orders"},
        "group_by": ["cid"],
        "aggregates": [{"fn": "sum", "column": "amt", "as": "total"}],
    }
    ev = make_evaluator(tables)
    rows = ev.eval(plan)
    cols = ev.schema(plan)
    g10 = next(r for r in rows if r["data"][cols.index("cid")] == 10)
    g20 = next(r for r in rows if r["data"][cols.index("cid")] == 20)
    assert g10["witnesses"] == {frozenset([L("orders", 1), L("orders", 2)])}
    assert g20["witnesses"] == {frozenset([L("orders", 3), L("orders", 4)])}
    # Killers never reference rows of other groups.
    for h in row_killers(g10):
        assert all(lab[0] == "orders" and lab[1][0] in (1, 2) for lab in h)


def test_join_then_aggregate_killers_agree_with_recompute():
    # Aggregate over a join: members have multi-row witnesses and the
    # evaluator validates provenance by sub-snapshot recomputation.
    specs = [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": ["oid", "cid", "amt"],
            "rows": [
                {"oid": 1, "cid": 10, "amt": 50},
                {"oid": 2, "cid": 10, "amt": 30},
                {"oid": 3, "cid": 20, "amt": 9},
            ],
        },
        {
            "table": "customers",
            "primary_key": ["id"],
            "columns": ["id"],
            "rows": [{"id": 10}, {"id": 20}],
        },
    ]
    plan = {
        "type": "aggregate",
        "input": {
            "type": "join",
            "left": {"type": "scan", "table": "orders"},
            "right": {"type": "scan", "table": "customers"},
            "on": [{"left": "cid", "right": "id"}],
        },
        "group_by": ["cid"],
        "aggregates": [
            {"fn": "count", "column": "*", "as": "n"},
            {"fn": "sum", "column": "amt", "as": "total"},
        ],
    }
    tables = {s["table"]: Table(s) for s in specs}
    ev = Evaluator(tables)
    rows = ev.eval(plan)
    cols = ev.schema(plan)
    g10 = next(r for r in rows if r["data"][cols.index("cid")] == 10)
    assert g10["data"] == (10, 2, 80)
    assert g10["witnesses"] == {
        frozenset([L("orders", 1), L("orders", 2), L("customers", 10)])
    }
    # Deleting the customer empties the group; deleting either order
    # changes count/sum. Every killer is validated against recomputation.
    assert set(row_killers(g10)) == {
        frozenset([L("customers", 10)]),
        frozenset([L("orders", 1)]),
        frozenset([L("orders", 2)]),
    }
    for h in row_killers(g10):
        by_table: dict = {}
        for lab in h:
            by_table.setdefault(lab[0], set()).add(lab[1])
        filtered = {
            s["table"]: Table({**s, "rows": [
                row for row in s["rows"]
                if tuple(row[c] for c in s["primary_key"])
                not in by_table.get(s["table"], set())
            ]})
            for s in specs
        }
        from app.errors import ApiError

        try:
            remain = Evaluator(filtered, with_provenance=False).eval(plan)
            assert all(rr["data"] != g10["data"] for rr in remain)
        except ApiError:
            pass  # aggregate input emptied -> target absent


def test_empty_aggregate_input():
    from app.errors import ApiError

    tables = [
        {
            "table": "t",
            "primary_key": ["id"],
            "columns": ["id", "v"],
            "rows": [{"id": 1, "v": 1}],
        }
    ]
    plan = {
        "type": "aggregate",
        "input": {
            "type": "filter",
            "input": {"type": "scan", "table": "t"},
            "predicate": {"column": "v", "op": "=", "value": 99},
        },
        "aggregates": [{"fn": "count", "column": "*", "as": "n"}],
    }
    ev = make_evaluator(tables)
    try:
        ev.eval(plan)
        assert False, "expected ApiError"
    except ApiError as e:
        assert e.code == "EMPTY_AGGREGATE"
