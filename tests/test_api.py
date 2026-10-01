"""HTTP-level tests using FastAPI TestClient."""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    os.environ["PROVENANCE_DB"] = tmp.name
    # Import after env is set so the store picks up the temp DB.
    import importlib

    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c
    main.store.close()
    os.unlink(tmp.name)


BASE_REQUEST = {
    "idempotency_key": "key-join-1",
    "tables": [
        {
            "table": "orders",
            "primary_key": ["oid"],
            "columns": [
                {"name": "oid", "type": "int"},
                {"name": "cust", "type": "int"},
                {"name": "amt", "type": "int"},
            ],
            "rows": [
                {"oid": 1, "cust": 10, "amt": 5},
                {"oid": 2, "cust": 10, "amt": 7},
                {"oid": 3, "cust": 20, "amt": 9},
            ],
        },
        {
            "table": "customers",
            "primary_key": ["id"],
            "columns": [
                {"name": "id", "type": "int"},
                {"name": "name", "type": "string"},
            ],
            "rows": [{"id": 10, "name": "A"}, {"id": 20, "name": "B"}],
        },
    ],
    "plan": {
        "type": "join",
        "left": {"type": "scan", "table": "orders"},
        "right": {"type": "scan", "table": "customers"},
        "on": [{"left": "cust", "right": "id"}],
    },
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.content == b'{"status":"ok"}'


def test_join_query_and_counterfactual(client):
    req = dict(BASE_REQUEST)
    req["explain"] = {"row": {"cust": 10, "oid": 1, "amt": 5, "id": 10, "name": "A"}}
    r = client.post("/query", json=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["row_count"] == 3
    cf = body["counterfactual"]
    # The joined output oid=1 has exactly one minimal witness, hence two
    # singleton minimal deletion sets.
    assert cf["minimal_deletion_count"] == 2
    deleted = {tuple(sorted(d)) for d in cf["minimal_deletion_sets"]}
    assert deleted == {("customers[10]",), ("orders[1]",)}
    for sc in cf["scenarios"]:
        assert sc["target_present"] is False
        # deleting customer 10 removes both oid=1,2 rows; deleting order 1 only one
        if sc["delete"] == ["customers[10]"]:
            assert sc["result"]["row_count"] == 1
        else:
            assert sc["result"]["row_count"] == 2


def test_idempotent_replay_byte_identical(client):
    req = {**BASE_REQUEST, "idempotency_key": "key-replay-1"}
    r1 = client.post("/query", json=req)
    r2 = client.post("/query", json=req)
    assert r1.status_code == 200
    assert r1.content == r2.content
    assert r2.headers["x-idempotent-replay"] == "true"


def test_idempotency_conflict(client):
    modified = dict(BASE_REQUEST)
    modified["idempotency_key"] = "key-conflict-1"
    r1 = client.post("/query", json=modified)
    assert r1.status_code == 200
    modified = {
        **modified,
        "tables": [
            {
                **BASE_REQUEST["tables"][0],
                "rows": BASE_REQUEST["tables"][0]["rows"]
                + [{"oid": 4, "cust": 20, "amt": 1}],
            },
            BASE_REQUEST["tables"][1],
        ],
    }
    r2 = client.post("/query", json=modified)
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "IDEMPOTENCY_KEY_CONFLICT"


def test_key_order_invariant(client):
    base = {**BASE_REQUEST, "idempotency_key": "key-shuffle-a"}
    r1 = client.post("/query", json=base)
    # same content, shuffled key order in a table row
    shuffled = {
        **BASE_REQUEST,
        "idempotency_key": "key-shuffle-b",
        "tables": [
            {
                **BASE_REQUEST["tables"][0],
                "rows": [
                    {"amt": 5, "cust": 10, "oid": 1},
                    {"oid": 2, "cust": 10, "amt": 7},
                    {"oid": 3, "cust": 20, "amt": 9},
                ],
            },
            BASE_REQUEST["tables"][1],
        ],
    }
    r2 = client.post("/query", json=shuffled)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.content == r2.content


def test_duplicate_primary_key(client):
    bad = {
        "idempotency_key": "key-dup-pk",
        "tables": [
            {
                "table": "t",
                "primary_key": ["id"],
                "columns": ["id"],
                "rows": [{"id": 1}, {"id": 1}],
            }
        ],
        "plan": {"type": "scan", "table": "t"},
    }
    r = client.post("/query", json=bad)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "DUPLICATE_PRIMARY_KEY"


def test_illegal_reference(client):
    bad = {
        "idempotency_key": "key-bad-ref",
        "tables": [
            {"table": "t", "primary_key": ["id"], "columns": ["id", "v"],
             "rows": [{"id": 1, "v": 1}]}
        ],
        "plan": {
            "type": "filter",
            "input": {"type": "scan", "table": "t"},
            "predicate": {"column": "nope", "op": "=", "value": 1},
        },
    }
    r = client.post("/query", json=bad)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "ILLEGAL_REFERENCE"


def test_unknown_table(client):
    bad = {
        "idempotency_key": "key-bad-table",
        "tables": [
            {"table": "t", "primary_key": ["id"], "columns": ["id"],
             "rows": [{"id": 1}]}
        ],
        "plan": {"type": "scan", "table": "other"},
    }
    r = client.post("/query", json=bad)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UNKNOWN_TABLE"


def test_sum_type_error(client):
    bad = {
        "idempotency_key": "key-sum-type",
        "tables": [
            {"table": "t", "primary_key": ["id"], "columns": ["id", "v"],
             "rows": [{"id": 1, "v": "x"}]}
        ],
        "plan": {
            "type": "aggregate",
            "input": {"type": "scan", "table": "t"},
            "aggregates": [{"fn": "sum", "column": "v", "as": "s"}],
        },
    }
    r = client.post("/query", json=bad)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "TYPE_ERROR"


def test_null_filter_3vl(client):
    req = {
        "idempotency_key": "key-null-3vl",
        "tables": [
            {
                "table": "t",
                "primary_key": ["id"],
                "columns": ["id", "v"],
                "rows": [{"id": 1, "v": 5}, {"id": 2, "v": None}, {"id": 3, "v": 9}],
            }
        ],
        "plan": {
            "type": "filter",
            "input": {"type": "scan", "table": "t"},
            "predicate": {"column": "v", "op": "<>", "value": 5},
        },
    }
    r = client.post("/query", json=req)
    assert r.status_code == 200
    rows = r.json()["result"]["rows"]
    assert [x["row"]["id"] for x in rows] == [3]


def test_aggregate_counterfactual_group_deletion(client):
    req = {
        "idempotency_key": "key-agg-cf",
        "tables": [
            {
                "table": "orders",
                "primary_key": ["oid"],
                "columns": ["oid", "cid", "amount"],
                "rows": [
                    {"oid": 1, "cid": 10, "amount": 5},
                    {"oid": 2, "cid": 10, "amount": 7},
                    {"oid": 3, "cid": 20, "amount": 9},
                ],
            }
        ],
        "plan": {
            "type": "aggregate",
            "input": {"type": "scan", "table": "orders"},
            "group_by": ["cid"],
            "aggregates": [{"fn": "sum", "column": "amount", "as": "total"}],
        },
        "explain": {"row": {"cid": 10, "total": 12}},
    }
    r = client.post("/query", json=req)
    assert r.status_code == 200, r.text
    cf = r.json()["counterfactual"]
    assert cf["target_provenance"]["dnf"] == "(s1) ∨ (s2)"
    # Group survives on either member, so both rows must be deleted.
    assert cf["minimal_deletion_sets"] == [["orders[1]", "orders[2]"]]
    scenario = cf["scenarios"][0]
    assert scenario["target_present"] is False
    assert scenario["result"]["rows"] == [{"cid": 20, "total": 9}]


def test_counterfactual_recompute_empty_aggregate(client):
    req = {
        "idempotency_key": "key-cf-empty-agg",
        "tables": [
            {"table": "t", "primary_key": ["id"], "columns": ["id", "v"],
             "rows": [{"id": 1, "v": 5}]}
        ],
        "plan": {
            "type": "aggregate",
            "input": {"type": "scan", "table": "t"},
            "aggregates": [{"fn": "count", "column": "*", "as": "n"}],
        },
        "explain": {"row": {"n": 1}},
    }
    r = client.post("/query", json=req)
    assert r.status_code == 200
    sc = r.json()["counterfactual"]["scenarios"][0]
    assert sc["target_present"] is False
    assert sc["recompute_error"]["code"] == "EMPTY_AGGREGATE"


def test_error_response_is_canonical_and_cached(client):
    req = {
        "idempotency_key": "key-err-cached",
        "tables": [
            {"table": "t", "primary_key": ["id"], "columns": ["id"],
             "rows": [{"id": 1}, {"id": 1}]}
        ],
        "plan": {"type": "scan", "table": "t"},
    }
    r1 = client.post("/query", json=req)
    r2 = client.post("/query", json=req)
    assert r1.status_code == 400
    assert r1.content == r2.content
    assert r2.headers["x-idempotent-replay"] == "true"
