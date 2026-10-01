"""Unit tests for minimal hitting sets and witness construction."""
from app.evaluator import Evaluator, Table
from app.provenance import minimal_hitting_sets


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
    # each single member is a minimal group witness
    assert g10["witnesses"] == {frozenset([L("orders", 1)]), frozenset([L("orders", 2)])}
    data = dict(zip(cols, g10["data"]))
    assert data == {"cust": 10, "n": 2, "total": 12, "lo": 5, "hi": 7}

    # To empty the group both members must go: the two singleton witnesses
    # form a hypergraph with a single minimal transversal {1, 2}.
    hs = minimal_hitting_sets(g10["witnesses"])
    assert hs == [frozenset([L("orders", 1), L("orders", 2)])]


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
