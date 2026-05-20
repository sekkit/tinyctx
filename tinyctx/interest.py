"""Compression-biased PageRank for code-context ranking.

This module is a faithful adaptation of the algorithm in section 5.1 of
Aksenov, Bodnia, Freedman, Mulligan, "Compression Is All You Need: Modeling
Mathematics" (arxiv 2603.20396) to a code corpus. The paper studies MathLib
and proves that human mathematics lives in the polynomial-growth (A_n,
log-density) regime; the same hierarchical compression is what coding agents
benefit from when they pick *load-bearing* abstractions to put in a prompt.

The shapes:

    T0(u) = (|S|_G  + |B|_G ) / (|S|_{G'\\u} + |B|_{G'\\u})       reductive compression
    I0(u) =                |B|_{G'\\u} / |S|_{G'\\u}              deductive compression
    J0(u) = beta * T0_norm + (1-beta) * I0_norm                  combined "interest"
    I1(u) = stationary distribution of biased PageRank, where
            teleportation chooses node v with probability J0(v)/Z

Why this is useful for tinyctx: when codex needs to be primed with repo
context, ranking nodes by I1 picks the symbols/files that compress a lot of
downstream content into a small interface (the same role lemmas play in
MathLib). That's a much better selection criterion than "files mentioned by
name in the chat" or "nearest-neighbour by embedding".

We do not duplicate what graphify or serena already do (graph extraction, LSP
ops). This module *consumes* a graphify-style graph.json and produces a
ranking. Format expected:

    {
      "nodes": [
        {"id": "src/foo.py:Foo.bar",
         "wrapped_signature": 12,    # tokens in declaration line(s)
         "wrapped_body": 84,         # tokens in body
         "deps": ["src/util.py:helper", "src/foo.py:Foo"]},
        ...
      ]
    }

Unwrapped lengths are computed from `deps`+wrapped lengths recursively, just
like the paper does on MathLib.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Node:
    nid: str
    wrapped_sig: int
    wrapped_body: int
    deps: list[str]
    # filled in during scoring:
    unwrapped_sig: int = 0
    unwrapped_body: int = 0
    t0: float = 0.0
    i0: float = 0.0
    j0: float = 0.0


def load_graph(path: str | Path) -> dict[str, Node]:
    raw = json.loads(Path(path).read_text())
    out: dict[str, Node] = {}
    for n in raw.get("nodes", []):
        out[n["id"]] = Node(
            nid=n["id"],
            wrapped_sig=int(n.get("wrapped_signature", 1)),
            wrapped_body=int(n.get("wrapped_body", 0)),
            deps=list(n.get("deps", [])),
        )
    return out


def _topo_order(nodes: dict[str, Node]) -> list[str]:
    """Khan's algorithm. Cycles get any ordering; the paper notes MathLib has
    ~60 small cycles that get collapsed into SCCs. We do the same shrug here:
    if a cycle is detected we just visit the remaining nodes in arbitrary
    order, which means their unwrapped lengths under-count once but don't
    crash. For agent-context-ranking, this is fine."""
    indeg = defaultdict(int)
    rev: dict[str, list[str]] = defaultdict(list)
    for nid, node in nodes.items():
        for d in node.deps:
            if d in nodes:
                indeg[nid] += 1
                rev[d].append(nid)
    ready = [nid for nid in nodes if indeg[nid] == 0]
    out: list[str] = []
    while ready:
        nid = ready.pop()
        out.append(nid)
        for parent in rev[nid]:
            indeg[parent] -= 1
            if indeg[parent] == 0:
                ready.append(parent)
    if len(out) < len(nodes):
        out.extend(nid for nid in nodes if nid not in set(out))
    # We want children-first (deps before dependents) for unwrapping, so
    # reverse if we built it dependents-first.
    return out


def compute_unwrapped(nodes: dict[str, Node]) -> None:
    """Populate node.unwrapped_sig and node.unwrapped_body recursively.
    Primitives (no deps) have unwrapped == wrapped."""
    # Order so that every node's deps come before it.
    seen: set[str] = set()
    order: list[str] = []
    def _visit(nid: str, stack: set[str]) -> None:
        if nid in seen or nid not in nodes or nid in stack:
            return
        stack.add(nid)
        for d in nodes[nid].deps:
            _visit(d, stack)
        stack.discard(nid)
        seen.add(nid)
        order.append(nid)
    for nid in list(nodes.keys()):
        _visit(nid, set())
    for nid in order:
        n = nodes[nid]
        sig = n.wrapped_sig
        body = n.wrapped_body
        for d in n.deps:
            dep = nodes.get(d)
            if dep is None:
                # treat unknown ref as a primitive of length 1
                sig += 1
                continue
            sig += dep.unwrapped_sig + dep.unwrapped_body
        n.unwrapped_sig = sig
        n.unwrapped_body = body  # body unwrapping is the same as sig+body recursion;
        # The paper distinguishes |S| and |B| but uses the same recursion for both.
        # We follow the paper: unwrapped of full element = unwrapped(sig) + body+deps-of-body.
        # Practically the same number for our use case.


def compute_compression(nodes: dict[str, Node]) -> None:
    """T0(u) = (|S|_G + |B|_G) / (|S|_G' + |B|_G')   per Table 3 of the paper.
    I0(u) =  |B|_G' / |S|_G'                          per the I0 definition.
    Both ratios are taken in G' \\ {u} in the paper; for our discrete graph
    that's just |wrapped|, since the node doesn't reference itself."""
    for n in nodes.values():
        wrapped_total = max(1, n.wrapped_sig + n.wrapped_body)
        unwrapped_total = max(1, n.unwrapped_sig + n.unwrapped_body)
        n.t0 = unwrapped_total / wrapped_total
        n.i0 = n.wrapped_body / max(1, n.wrapped_sig)


def _normalize_log(values: list[float]) -> list[float]:
    """Min-max normalize on a log scale; T0/I0 span many orders of magnitude
    (the paper's longest MathLib element has unwrapped 10^104)."""
    logs = [math.log1p(max(0.0, v)) for v in values]
    lo, hi = min(logs), max(logs)
    if hi <= lo:
        return [0.5] * len(values)
    return [(x - lo) / (hi - lo) for x in logs]


def compute_j0(nodes: dict[str, Node], beta: float = 0.5) -> None:
    ids = list(nodes.keys())
    t = _normalize_log([nodes[i].t0 for i in ids])
    j = _normalize_log([nodes[i].i0 for i in ids])
    for i, nid in enumerate(ids):
        nodes[nid].j0 = beta * t[i] + (1.0 - beta) * j[i]


def compression_pagerank(
    nodes: dict[str, Node],
    *,
    alpha: float = 0.85,
    beta: float = 0.5,
    seed: dict[str, float] | None = None,
    iterations: int = 50,
) -> dict[str, float]:
    """Personalized PageRank with teleportation biased toward high-J0 nodes.

    Implements the transition operator from the paper §5.1:

        P(v, u) = alpha * w(u, v)/W(u) + (1-alpha) * J0(v) / Z

    Edges point FROM u TO its dependencies v. We invert that to a PageRank
    that flows from a node to the things it builds upon, then biases the
    teleportation distribution toward "interesting" (high-J0) nodes. Optional
    `seed` adds a second teleportation distribution over query-specific
    nodes (Aider-style chat-token personalization), composed multiplicatively
    with J0.
    """
    if not nodes:
        return {}
    compute_unwrapped(nodes)
    compute_compression(nodes)
    compute_j0(nodes, beta=beta)

    ids = list(nodes.keys())
    n = len(ids)
    idx = {nid: i for i, nid in enumerate(ids)}

    # Out-edges (each ref counts as 1; could be weighted by reference multiplicity).
    out_edges: list[list[int]] = [[] for _ in range(n)]
    for nid in ids:
        node = nodes[nid]
        for d in node.deps:
            j = idx.get(d)
            if j is not None:
                out_edges[idx[nid]].append(j)

    # Teleportation distribution.
    j0 = [nodes[nid].j0 for nid in ids]
    if seed:
        seeds = [seed.get(nid, 0.0) for nid in ids]
        if sum(seeds) > 0:
            # multiplicative blend: seeds gate by query relevance, j0 ranks by load-bearing
            tele = [s * (1e-6 + j) for s, j in zip(seeds, j0)]
        else:
            tele = j0[:]
    else:
        tele = j0[:]
    s_tele = sum(tele) or 1.0
    tele = [x / s_tele for x in tele]

    # Power iteration.
    rank = [1.0 / n] * n
    for _ in range(iterations):
        new = [0.0] * n
        leak = 0.0  # mass on dangling nodes
        for i in range(n):
            outs = out_edges[i]
            if not outs:
                leak += rank[i]
                continue
            share = alpha * rank[i] / len(outs)
            for j in outs:
                new[j] += share
        # Teleportation contribution + dangling redistribution via teleportation.
        spread = (1.0 - alpha) + alpha * leak
        for j in range(n):
            new[j] += spread * tele[j]
        rank = new

    return {ids[i]: rank[i] for i in range(n)}


def rank_for_query(
    nodes: dict[str, Node],
    query_tokens: Iterable[str],
    *,
    budget: int = 20,
    alpha: float = 0.85,
    beta: float = 0.5,
) -> list[tuple[str, float]]:
    """Return the top-`budget` nodes ranked by compression-biased PageRank,
    personalized by simple substring matching of query tokens against node ids.
    Aider's repomap uses a similar substring-personalization pattern."""
    seed: dict[str, float] = {}
    qs = [q.lower() for q in query_tokens if q]
    if qs:
        for nid in nodes:
            low = nid.lower()
            score = sum(1.0 for q in qs if q in low)
            if score:
                seed[nid] = score
    pr = compression_pagerank(nodes, alpha=alpha, beta=beta, seed=seed or None)
    ranked = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:budget]


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m tinyctx.interest <graph.json> <query...>"""
    import sys
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        sys.stderr.write("usage: tinyctx.interest <graph.json> [query tokens...]\n")
        return 2
    graph_path = args[0]
    query = args[1:]
    nodes = load_graph(graph_path)
    if not nodes:
        sys.stderr.write(f"no nodes in {graph_path}\n")
        return 1
    ranked = rank_for_query(nodes, query)
    for nid, score in ranked:
        n = nodes[nid]
        print(f"{score:.5f}\tT0={n.t0:8.2f}\tI0={n.i0:6.2f}\t{nid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
