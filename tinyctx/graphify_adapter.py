"""Adapter from graphify's exported graph.json to the shape consumed by
tinyctx.interest.

graphify (safishamsi/graphify) exports an HTML wiki and a graph.json. The
graph.json shape varies between versions but is consistently one of two
families: a node-link form (NetworkX-compatible) or a flat {nodes, edges}
form. We accept both; if neither matches, we fall back to a permissive
recursive walker that infers id/text fields.

Output shape (consumed by tinyctx/interest.py):

    {
      "nodes": [
        {"id": str,
         "wrapped_signature": int,   # tokens in declaration
         "wrapped_body": int,        # tokens in body
         "deps": [str, ...]}         # ids of nodes this one references
      ]
    }

Wrapped lengths are approximated from the source text graphify already
extracted (it captures function/class bodies as `code` or `text` per node).
We split on whitespace; this is a slight overestimate of token count but
the shape of the analysis only depends on relative magnitudes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


_TOKEN_SPLIT = re.compile(r"\s+|(?=[^\w])|(?<=[^\w])")


def _approx_tokens(text: str | None) -> int:
    if not text:
        return 0
    parts = [p for p in _TOKEN_SPLIT.split(text) if p and not p.isspace()]
    return max(1, len(parts))


def _split_signature_body(text: str | None) -> tuple[int, int]:
    """For function/method nodes, the first line is roughly the signature;
    the rest is the body. For other nodes, treat everything as signature."""
    if not text:
        return 1, 0
    head, _, rest = text.partition("\n")
    sig = _approx_tokens(head)
    body = _approx_tokens(rest) if rest else 0
    return sig, body


def _extract_id(node: dict[str, Any]) -> str | None:
    for key in ("id", "name", "qualified_name", "fullname", "path"):
        v = node.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _extract_text(node: dict[str, Any]) -> str:
    for key in ("code", "text", "snippet", "summary", "body", "content", "label"):
        v = node.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _extract_deps(node: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("deps", "dependencies", "references", "imports", "calls", "neighbors"):
        v = node.get(key)
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict):
                    nid = _extract_id(x)
                    if nid:
                        out.append(nid)
    return out


def _from_node_link(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """NetworkX `json_graph.node_link_data` form: {nodes:[...], links:[...]}.
    Build deps from the link list."""
    nodes = graph.get("nodes", [])
    links = graph.get("links") or graph.get("edges") or []
    deps_by: dict[str, list[str]] = {}
    for link in links:
        if not isinstance(link, dict):
            continue
        s = link.get("source")
        t = link.get("target")
        if isinstance(s, dict):
            s = _extract_id(s)
        if isinstance(t, dict):
            t = _extract_id(t)
        if isinstance(s, str) and isinstance(t, str):
            deps_by.setdefault(s, []).append(t)
    out: list[dict[str, Any]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = _extract_id(n)
        if not nid:
            continue
        sig, body = _split_signature_body(_extract_text(n))
        deps = _extract_deps(n) or deps_by.get(nid, [])
        out.append({"id": nid, "wrapped_signature": sig, "wrapped_body": body, "deps": deps})
    return out


def _from_flat(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """{nodes: [{id, ..., deps:[...]}]} form — already close to ours."""
    out: list[dict[str, Any]] = []
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        nid = _extract_id(n)
        if not nid:
            continue
        sig, body = _split_signature_body(_extract_text(n))
        deps = _extract_deps(n)
        out.append({"id": nid, "wrapped_signature": sig, "wrapped_body": body, "deps": deps})
    return out


def adapt(graphify_graph: dict[str, Any]) -> dict[str, Any]:
    """Detect shape and convert. Always returns the tinyctx-shaped graph."""
    if "links" in graphify_graph or any(
        isinstance(l, dict) and "source" in l and "target" in l
        for l in graphify_graph.get("edges", []) if isinstance(l, dict)
    ):
        nodes = _from_node_link(graphify_graph)
    else:
        nodes = _from_flat(graphify_graph)

    # Fallback: if nothing was found, do a recursive walk for any list of
    # dicts under "nodes"-like keys.
    if not nodes:
        nodes = _from_flat(graphify_graph)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for n in nodes:
        if n["id"] in seen:
            continue
        seen.add(n["id"])
        deduped.append(n)
    return {"nodes": deduped}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        sys.stderr.write(
            "usage: python -m tinyctx.graphify_adapter <graphify-out.json> [--out tinyctx-graph.json]\n"
        )
        return 2
    src = Path(args[0])
    out_path = None
    if "--out" in args:
        idx = args.index("--out")
        if idx + 1 < len(args):
            out_path = Path(args[idx + 1])
    raw = json.loads(src.read_text())
    converted = adapt(raw)
    text = json.dumps(converted, indent=2)
    if out_path:
        out_path.write_text(text)
        sys.stderr.write(f"wrote {len(converted['nodes'])} nodes to {out_path}\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
