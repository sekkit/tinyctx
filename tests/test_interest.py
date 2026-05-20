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
    _normalize_log,
    compression_pagerank,
    compute_compression,
    compute_j0,
    compute_unwrapped,
    load_graph,
    main,
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


def test_pagerank_empty_graph_returns_empty_dict():
    assert compression_pagerank({}) == {}


def test_pagerank_single_node_gets_full_mass():
    only = Node("solo", wrapped_sig=3, wrapped_body=5, deps=[])
    nodes = _g(only)
    pr = compression_pagerank(nodes, iterations=20)
    assert set(pr) == {"solo"}
    assert abs(pr["solo"] - 1.0) < 1e-9


def test_pagerank_is_deterministic():
    """Same input must produce same ranks (no RNG, no dict-iteration entropy)."""
    p = Node("p", 1, 0, [])
    a = Node("a", 2, 0, ["p", "p"])
    b = Node("b", 2, 0, ["a"])
    c = Node("c", 2, 0, ["a", "b"])
    nodes_a = _g(p, a, b, c)
    nodes_b = _g(p, a, b, c)
    # compression_pagerank mutates nodes; rebuild to be fair.
    pr1 = compression_pagerank(nodes_a, iterations=40)
    pr2 = compression_pagerank(nodes_b, iterations=40)
    assert pr1 == pr2


def test_normalize_log_all_equal_returns_half():
    """When max == min, normalization can't divide by zero; spec is to return 0.5."""
    assert _normalize_log([0.0, 0.0, 0.0]) == [0.5, 0.5, 0.5]
    assert _normalize_log([7.5, 7.5]) == [0.5, 0.5]


def test_normalize_log_min_max_endpoints():
    """Smallest input maps to 0.0; largest maps to 1.0."""
    out = _normalize_log([0.0, 1.0, 1000.0])
    assert out[0] == 0.0
    assert out[-1] == 1.0
    # middle value strictly between endpoints
    assert 0.0 < out[1] < 1.0


def test_compute_unwrapped_handles_cycles_without_crashing():
    """Cycles are tolerated; the module's docstring acknowledges this."""
    a = Node("a", 1, 0, ["b"])
    b = Node("b", 1, 0, ["c"])
    c = Node("c", 1, 0, ["a"])
    nodes = _g(a, b, c)
    compute_unwrapped(nodes)  # must not raise / recurse forever
    # every node got a non-zero unwrapped_sig
    for nid in ("a", "b", "c"):
        assert nodes[nid].unwrapped_sig >= 1


def test_compute_unwrapped_unknown_dep_treated_as_primitive():
    a = Node("a", 1, 0, ["nope_does_not_exist"])
    nodes = _g(a)
    compute_unwrapped(nodes)
    # wrapped_sig=1 + 1 (unknown dep treated as primitive of length 1)
    assert nodes["a"].unwrapped_sig == 2


def test_beta_extremes_change_j0_weighting():
    """beta=1.0 -> j0 == normalized(t0); beta=0.0 -> j0 == normalized(i0)."""
    p = Node("p", 1, 0, [])
    deep_chain = Node("deep", 2, 0, ["p"] * 20)        # high T0, modest I0
    fat_body = Node("fat", 2, 200, [])                  # tiny T0, high I0
    nodes = _g(p, deep_chain, fat_body)
    compute_unwrapped(nodes)
    compute_compression(nodes)
    # beta=1: j0 dominated by T0 normalization, so 'deep' ranks above 'fat'
    compute_j0(nodes, beta=1.0)
    j_deep_t = nodes["deep"].j0
    j_fat_t = nodes["fat"].j0
    # beta=0: j0 dominated by I0 normalization, so 'fat' ranks above 'deep'
    compute_j0(nodes, beta=0.0)
    j_deep_i = nodes["deep"].j0
    j_fat_i = nodes["fat"].j0
    assert j_deep_t > j_fat_t, f"beta=1 should favour deep: {j_deep_t} vs {j_fat_t}"
    assert j_fat_i > j_deep_i, f"beta=0 should favour fat: {j_fat_i} vs {j_deep_i}"


def test_load_graph_default_field_values():
    """Missing wrapped_signature/wrapped_body/deps fall back to documented defaults."""
    with TemporaryDirectory() as td:
        path = Path(td) / "graph.json"
        path.write_text(json.dumps({"nodes": [{"id": "minimal"}]}))
        nodes = load_graph(str(path))
        assert nodes["minimal"].wrapped_sig == 1   # default wrapped_signature
        assert nodes["minimal"].wrapped_body == 0
        assert nodes["minimal"].deps == []


def test_rank_for_query_empty_query_falls_back_to_unbiased():
    """No query tokens => no seed personalization, but ranking still works."""
    p = Node("p", 1, 0, [])
    a = Node("a", 2, 0, ["p"])
    b = Node("b", 2, 0, ["a"])
    nodes = _g(p, a, b)
    ranked = rank_for_query(nodes, [], budget=3)
    assert len(ranked) == 3
    # ranks form a probability distribution (no seed -> still teleports via J0)
    total = sum(score for _, score in ranked)
    assert 0.99 < total < 1.01


def test_rank_for_query_filters_blank_strings():
    """Empty/whitespace tokens are dropped; only non-empty strings personalize."""
    p = Node("p", 1, 0, [])
    a = Node("alpha_node", 2, 0, ["p"] * 5)
    b = Node("beta_node", 2, 0, ["p"] * 5)
    nodes = _g(p, a, b)
    # Empty strings get filtered out by `if q` so this behaves like no-query.
    ranked_blank = rank_for_query(nodes, ["", "", ""], budget=3)
    ranked_none = rank_for_query(nodes, [], budget=3)
    # rebuild fresh nodes since compression_pagerank mutates
    p2 = Node("p", 1, 0, [])
    a2 = Node("alpha_node", 2, 0, ["p"] * 5)
    b2 = Node("beta_node", 2, 0, ["p"] * 5)
    nodes2 = _g(p2, a2, b2)
    ranked_none = rank_for_query(nodes2, [], budget=3)
    assert dict(ranked_blank) == dict(ranked_none)


def test_main_cli_no_args_returns_exit_code_2(capsys):
    rc = main([])
    err = capsys.readouterr().err
    assert rc == 2
    assert "usage" in err.lower()


def test_main_cli_with_empty_graph_returns_1(capsys, tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"nodes": []}))
    rc = main([str(p)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no nodes" in err.lower()


def test_main_cli_runs_and_prints_ranking(capsys, tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({
        "nodes": [
            {"id": "src/a.py:foo", "wrapped_signature": 3, "wrapped_body": 0, "deps": ["b"]},
            {"id": "b", "wrapped_signature": 1, "wrapped_body": 0, "deps": []},
        ]
    }))
    rc = main([str(p), "foo"])
    out = capsys.readouterr().out
    assert rc == 0
    # Both node ids appear, query-personalized one ranks high.
    assert "src/a.py:foo" in out
    assert "T0=" in out and "I0=" in out


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
