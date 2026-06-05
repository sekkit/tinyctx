"""AST splittable-node-types table — a portable primitive for tree-sitter
based code chunking.

This module ships **data** plus a small pure helper. It deliberately does
not depend on the ``tree_sitter`` Python bindings: tinyctx already
delegates tree-sitter parsing to graphify (codex skill) and gitnexus
(MCP server), and we don't want to vendor a second tree-sitter
installation.

Borrowed-as-data (not code) from zilliztech/claude-context's
``packages/core/src/splitter/ast-splitter.ts:16-26``. Original list
maintained by Cheney Zhang et al, MIT-licensed at
<https://github.com/zilliztech/claude-context>. Reproduced here verbatim
under MIT to keep the table reusable inside tinyctx without taking a
runtime dependency on the upstream.

Two consumer shapes are supported:

  1. **Direct tree-sitter integration**: callers that already have a
     parsed AST (raw tree-sitter Node objects) match ``node.type`` against
     :data:`SPLITTABLE_NODE_TYPES[lang]` and slice ``code[node.start_byte:
     node.end_byte]`` to materialize chunks. claude-context's
     `extractChunks` walks the tree depth-first, emitting a chunk for
     *every* matching node — i.e. classes and the methods they contain
     are both emitted, with overlapping ranges deduped at retrieval time.

  2. **graphify graph.json reuse**: tinyctx already runs graphify per
     project, which dumps a code-graph into ``graphify-out/graph.json``.
     :func:`chunks_from_graphify_graph` walks that JSON and emits
     :class:`Chunk` records without re-parsing.

     **Caveat** — graphify's current schema (verified 2026-05) keeps only
     ``source_file`` + ``source_location='L<n>'`` (start line only) +
     ``file_type ∈ {code, document, rationale}``. It uses tree-sitter at
     extraction time but does not preserve AST node types in the graph.
     So Chunks produced from a graphify graph have ``node_type=None`` and
     ``end_line=start_line``. Filtering by :data:`SPLITTABLE_NODE_TYPES`
     therefore requires a richer source — either tree-sitter directly or
     a graph producer (e.g. gitnexus, future graphify) that preserves the
     AST type per node.

The table is currently primitive-only — no consumer in tinyctx wires it
into the wire pipeline yet. It exists so that:

  * future scout enhancements can produce AST-aligned previews instead
    of head-of-file slices;
  * users who run an external semantic-RAG MCP (e.g. claude-context
    itself, configured to talk to local Milvus + Ollama) can reuse the
    same chunk boundaries tinyctx would have used;
  * tests / external tools can validate the table against new
    tree-sitter grammar releases without crawling claude-context's TS.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# ============================================================================
# The table — verbatim from claude-context, MIT-licensed.
# Source: packages/core/src/splitter/ast-splitter.ts:16-26 @ commit master.
# Last cross-checked: 2026-05-10.
# ============================================================================

SPLITTABLE_NODE_TYPES: dict[str, frozenset[str]] = {
    "javascript": frozenset({
        "function_declaration", "arrow_function", "class_declaration",
        "method_definition", "export_statement",
    }),
    "typescript": frozenset({
        "function_declaration", "arrow_function", "class_declaration",
        "method_definition", "export_statement",
        "interface_declaration", "type_alias_declaration",
    }),
    "python": frozenset({
        "function_definition", "class_definition",
        "decorated_definition", "async_function_definition",
    }),
    "java": frozenset({
        "method_declaration", "class_declaration",
        "interface_declaration", "constructor_declaration",
    }),
    "cpp": frozenset({
        "function_definition", "class_specifier",
        "namespace_definition", "declaration",
    }),
    "go": frozenset({
        "function_declaration", "method_declaration",
        "type_declaration", "var_declaration", "const_declaration",
    }),
    "rust": frozenset({
        "function_item", "impl_item", "struct_item",
        "enum_item", "trait_item", "mod_item",
    }),
    "csharp": frozenset({
        "method_declaration", "class_declaration",
        "interface_declaration", "struct_declaration",
        "enum_declaration",
    }),
    "scala": frozenset({
        "method_declaration", "class_declaration",
        "interface_declaration", "constructor_declaration",
    }),
}
"""Per-language tree-sitter node types that should become their own chunk.
Keys match graphify / gitnexus language tags (lowercase, no leading dot).
``frozenset`` so callers can't accidentally mutate the canonical table."""

LANGUAGE_ALIASES: dict[str, str] = {
    "js": "javascript", "mjs": "javascript", "cjs": "javascript",
    "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "py": "python", "pyi": "python",
    "c++": "cpp", "cxx": "cpp", "cc": "cpp", "c": "cpp",
    "h": "cpp", "hpp": "cpp", "hh": "cpp",
    "rs": "rust",
    "cs": "csharp",
}
"""Common file-extension and casual aliases mapped to canonical keys of
:data:`SPLITTABLE_NODE_TYPES`. Mirrors claude-context's
``getLanguageConfig`` mapping (`ast-splitter.ts:86-107`)."""


def canonical_lang(name: str) -> str | None:
    """Normalize a language tag to a key in :data:`SPLITTABLE_NODE_TYPES`,
    or return ``None`` if unsupported. Accepts mixed case and known
    aliases (``js`` → ``javascript`` etc.)."""
    if not name:
        return None
    key = name.strip().lstrip(".").lower()
    if key in SPLITTABLE_NODE_TYPES:
        return key
    return LANGUAGE_ALIASES.get(key)


def is_splittable(lang: str, node_type: str) -> bool:
    """True iff ``node_type`` should become a chunk for the given language."""
    canon = canonical_lang(lang)
    if canon is None:
        return False
    return node_type in SPLITTABLE_NODE_TYPES[canon]


# ============================================================================
# Chunk records + graphify-graph adapter
# ============================================================================

@dataclass(frozen=True)
class Chunk:
    """A single AST-aligned slice of source. Lines are 1-based inclusive,
    matching claude-context's Milvus schema fields ``startLine`` /
    ``endLine`` (`milvus-vectordb.ts:215-274`).

    ``node_type`` is ``None`` when the source graph does not preserve
    tree-sitter AST node types (current graphify behaviour); callers
    that want to filter against :data:`SPLITTABLE_NODE_TYPES` should
    treat ``None`` as "unknown — pass through".
    """

    file: str
    language: str | None
    node_type: str | None
    start_line: int
    end_line: int
    symbol: str | None = None
    """Best-effort enclosing symbol name (function / class / etc.). May be
    ``None`` when the source graph doesn't expose one."""


_EXT_TO_LANG = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".py": "python", ".pyi": "python",
    ".java": "java",
    ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp",
    ".c": "cpp", ".h": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".scala": "scala",
}


def _lang_from_path(path: str) -> str | None:
    if not path:
        return None
    dot = path.rfind(".")
    if dot < 0:
        return None
    return _EXT_TO_LANG.get(path[dot:].lower())


def _parse_source_location(loc: Any) -> tuple[int | None, int | None]:
    """Parse graphify's ``source_location`` field. Known shapes:
      * ``"L14"`` — start line only (graphify ≤ 0.x current behaviour)
      * ``"L14-25"`` — start-end (older graphify variants)
      * ``"L14:5"`` — start line + col (drop col)
      * int — already a start line
    Returns ``(start_line, end_line)`` with either or both ``None``
    when the shape is unrecognised.
    """
    if isinstance(loc, int):
        return loc, loc
    if not isinstance(loc, str) or not loc:
        return None, None
    s = loc.strip().lstrip("Ll")
    if "-" in s:
        a, _, b = s.partition("-")
        try:
            return int(a), int(b)
        except ValueError:
            return None, None
    if ":" in s:
        s = s.split(":", 1)[0]
    try:
        n = int(s)
        return n, n
    except ValueError:
        return None, None


def chunks_from_graphify_graph(graph: dict[str, Any]) -> list[Chunk]:
    """Walk a code-graph JSON and emit one :class:`Chunk` per code node.

    Tolerant of three shapes:

    1. **graphify** (current, 2026-05): nodes carry ``source_file`` +
       ``source_location='L<n>'`` + ``file_type='code'``. AST node type
       is not preserved — emitted Chunks have ``node_type=None``.
       Non-code nodes (``file_type='rationale'``, ``'document'``) are
       skipped.
    2. **richer / hypothetical**: nodes carry explicit ``language``,
       ``node_type`` (or ``ast_type``), ``file``/``path``, ``start_line``,
       ``end_line``. When ``node_type`` is in :data:`SPLITTABLE_NODE_TYPES`
       for the resolved language, emit a Chunk. Other nodes are skipped.
    3. **gitnexus / nested-data**: same fields as (2) but lifted under a
       ``data`` sub-dict.

    Files with unrecognised extensions (no matching language) are still
    emitted in shape (1) — language defaults to ``None``. The function
    never raises on shape variance.
    """
    out: list[Chunk] = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        data = node.get("data") if isinstance(node.get("data"), dict) else {}

        # Shape (2)/(3): explicit AST type present → SPLITTABLE filter.
        node_type = (node.get("node_type") or node.get("ast_type")
                     or data.get("node_type") or data.get("ast_type"))
        if node_type:
            lang = (node.get("language") or data.get("language")
                    or node.get("lang") or data.get("lang"))
            canon = canonical_lang(lang or "")
            if canon is None or node_type not in SPLITTABLE_NODE_TYPES[canon]:
                continue
            file = node.get("file") or data.get("file") or data.get("path")
            start = node.get("start_line") or data.get("start_line")
            end = node.get("end_line") or data.get("end_line")
            if not (file and start and end):
                continue
            symbol = (node.get("name") or node.get("symbol")
                      or data.get("name") or data.get("symbol"))
            try:
                out.append(Chunk(
                    file=str(file),
                    language=canon,
                    node_type=str(node_type),
                    start_line=int(start),
                    end_line=int(end),
                    symbol=str(symbol) if symbol else None,
                ))
            except (TypeError, ValueError):
                continue
            continue  # don't double-process

        # Shape (1): graphify's current schema.
        file_type = node.get("file_type") or data.get("file_type")
        if file_type and file_type != "code":
            # Skip rationale / document / etc.
            continue
        file = (node.get("source_file") or node.get("file")
                or data.get("source_file") or data.get("file"))
        loc = (node.get("source_location") or node.get("location")
               or data.get("source_location") or data.get("location"))
        if not file:
            continue
        start_line, end_line = _parse_source_location(loc)
        if start_line is None:
            continue
        if end_line is None:
            end_line = start_line
        lang = _lang_from_path(str(file))
        symbol = (node.get("label") or node.get("name")
                  or data.get("label") or data.get("name"))
        out.append(Chunk(
            file=str(file),
            language=lang,
            node_type=None,
            start_line=start_line,
            end_line=end_line,
            symbol=str(symbol) if symbol else None,
        ))
    return out


def supported_languages() -> list[str]:
    """Sorted list of canonical language keys."""
    return sorted(SPLITTABLE_NODE_TYPES.keys())


# ============================================================================
# CLI
# ============================================================================

def _cmd_languages(_args: argparse.Namespace) -> int:
    for lang in supported_languages():
        types = sorted(SPLITTABLE_NODE_TYPES[lang])
        print(f"{lang:<12} {len(types):>2} types  " + ", ".join(types))
    return 0


def _cmd_chunks(args: argparse.Namespace) -> int:
    p = Path(args.graph)
    if not p.is_file():
        print(f"no such file: {p}", file=sys.stderr)
        return 1
    try:
        graph = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"failed to read {p}: {e}", file=sys.stderr)
        return 1
    chunks = chunks_from_graphify_graph(graph)
    if args.json:
        print(json.dumps([c.__dict__ for c in chunks], indent=2))
    else:
        for c in chunks:
            sym = f" {c.symbol}" if c.symbol else ""
            tag_l = c.language or "?"
            tag_t = c.node_type or "?"
            print(f"{c.file}:{c.start_line}-{c.end_line} "
                  f"[{tag_l}/{tag_t}]{sym}")
    if args.summary and not args.json:
        print(f"-- {len(chunks)} chunks across "
              f"{len({c.file for c in chunks})} files", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tinyctx-chunker",
        description=("Walk a graphify-style graph.json and emit AST-aligned "
                     "code chunks per claude-context's SPLITTABLE_NODE_TYPES "
                     "table. Pure data primitive — no tree-sitter dependency."),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("languages",
                        help="list supported language tags + node types")
    pl.set_defaults(func=_cmd_languages)

    pc = sub.add_parser("chunks",
                        help="emit chunks from a graphify graph.json")
    pc.add_argument("graph", help="path to graph.json (graphify or gitnexus)")
    pc.add_argument("--json", action="store_true",
                    help="emit chunks as a JSON array (default: text lines)")
    pc.add_argument("--summary", action="store_true",
                    help="append a summary line (text mode only)")
    pc.set_defaults(func=_cmd_chunks)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
