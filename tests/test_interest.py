"""Tests for tinyctx.interest. Builds tiny synthetic graphs and verifies:

  - T0 increases with hierarchical nesting (paper's main empirical claim
    on MathLib: log unwrapped grows linearly with depth).
  - I0 picks out terse-statement / long-body nodes.
  - Compression-biased PageRank gives more mass to a high-J0 utility node
    than to a leaf primitive, even when both have the same in-degree.
  - PageRank distribution sums to ~1.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tinyctx.interest import (
    compression_pagerank,
    compute_compression,
    compute_j0,
    compute_unwrapped,
    load_graph,
    rank_for_query,
    Node,
)


def _g(*nodes):
    return {n.nid: n for n in nodes}


def test_unwrapped_grows_with_depth():
    primitive = Node("p", wrapped_sig=1, wrapped_body=0, deps=[])
    lemma1   = Node("L1", wrapped_sig=2, wrapped_body=0, deps=["p", "p", "p"])
    lemma2   = Node("L2", wrapped_sig=2, wrapped_body=0, deps=["L1", "L1", "L1"])
    theorem  = Node("T",  wrapped_sig=2, wrapped_body=0, deps=["L2", "L2", "L2"])
    nodes = _g(primitive, lemma1, lemma2, theorem)

    compute_unwrapped(nodes)
    # primitive expands to itself (1 token); each level multiplies by 3 plus its own sig.
    assert nodes["p"].unwrapped_sig == 1
    assert nodes["L1"].unwrapped_sig > nodes["p"].unwrapped_sig
    assert nodes["L2"].unwrapped_sig > nodes["L1"].unwrapped_sig
    assert nodes["T"].unwrapped_sig  > nodes["L2"].unwrapped_sig
    # exponential, not linear (the paper's "log unwrapped grows ~linear with depth")
    ratio = nodes["T"].unwrapped_sig / max(1, nodes["L1"].unwrapped_sig)
    assert ratio >= 4, f"expected exponential growth across 2 levels, got ratio={ratio}"


def test_t0_rewards_compression():
    # node with deep deps but tiny wrapped form -> high T0.
    primitive = Node("p", wrapped_sig=1, wrapped_body=0, deps=[])
    deep = Node("deep", wrapped_sig=2, wrapped_body=0, deps=["p"] * 10)
    facade = Node("facade", wrapped_sig=2, wrapped_body=1, deps=["deep"])
    flat = Node("flat", wrapped_sig=20, wrapped_body=10, deps=["p"] * 8)
    nodes = _g(primitive, deep, facade, flat)
    compute_unwrapped(nodes)
    compute_compression(nodes)
    assert nodes["facade"].t0 > nodes["flat"].t0, \
        f"facade T0={nodes['facade'].t0} should beat flat T0={nodes['flat'].t0}"


def test_i0_rewards_short_signature_long_body():
    short_long = Node("FLT", wrapped_sig=4, wrapped_body=2000, deps=[])
    long_long  = Node("verbose", wrapped_sig=200, wrapped_body=2000, deps=[])
    short_short = Node("triv", wrapped_sig=4, wrapped_body=4, deps=[])
    nodes = _g(short_long, long_long, short_short)
    compute_unwrapped(nodes)
    compute_compression(nodes)
    assert nodes["FLT"].i0 > nodes["verbose"].i0
    assert nodes["FLT"].i0 > nodes["triv"].i0


def test_pagerank_sums_to_one():
    p = Node("p", 1, 0, [])
    a = Node("a", 2, 0, ["p"])
    b = Node("b", 2, 0, ["a"])
    c = Node("c", 2, 0, ["a", "b"])
    nodes = _g(p, a, b, c)
    pr = compression_pagerank(nodes, iterations=80)
    s = sum(pr.values())
    assert 0.99 < s < 1.01, f"PageRank should be a distribution, got sum={s}"


def test_biased_pagerank_lifts_load_bearing_nodes():
    """Build a graph where 'util' is referenced by many things and itself
    compresses many primitives. Biased PageRank should give it more mass than
    a leaf primitive that's referenced equally often."""
    p = Node("p", 1, 0, [])
    util = Node("util", 2, 0, ["p"] * 30)        # high T0: collapses 30 primitives
    callers = [Node(f"c{i}", 2, 0, ["util"]) for i in range(5)]
    leaf_referenced = Node("leaf", 1, 0, [])
    callers_to_leaf = [Node(f"d{i}", 2, 0, ["leaf"]) for i in range(5)]
    nodes = _g(p, util, leaf_referenced, *callers, *callers_to_leaf)
    pr = compression_pagerank(nodes, iterations=80)
    assert pr["util"] > pr["leaf"], (
        f"util (high T0) should rank above leaf primitive: "
        f"util={pr['util']:.4f} leaf={pr['leaf']:.4f}"
    )


def test_rank_for_query_substring_personalization():
    p = Node("p", 1, 0, [])
    auth = Node("src/auth.py:verify_token", 4, 50, ["p"] * 10)
    db = Node("src/db.py:execute", 4, 80, ["p"] * 12)
    util = Node("src/util.py:helper", 4, 5, ["p"] * 2)
    nodes = _g(p, auth, db, util)
    ranked = rank_for_query(nodes, ["auth", "token"], budget=3)
    top_ids = [nid for nid, _ in ranked]
    assert top_ids[0] == "src/auth.py:verify_token", f"ranked={ranked}"


def test_load_graph_roundtrip():
    with TemporaryDirectory() as td:
        path = Path(td) / "graph.json"
        path.write_text(json.dumps({
            "nodes": [
                {"id": "a", "wrapped_signature": 3, "wrapped_body": 0, "deps": ["b"]},
                {"id": "b", "wrapped_signature": 1, "wrapped_body": 0, "deps": []},
            ]
        }))
        nodes = load_graph(str(path))
        assert set(nodes) == {"a", "b"}
        assert nodes["a"].deps == ["b"]


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
