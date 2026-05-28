"""Local-docs MCP server: zero-network library introspection.

The offline replacement for context7. Exposes Python's already-installed
package metadata (`importlib.metadata`), rendered docs (`pydoc`), and
symbol-level source/docstrings (`inspect`) over the MCP stdio protocol so
codex can ask "what does X do?" without phoning home to a third party.

Hard limits — these are the cost of being truly offline:
  - Python packages only. Other languages need their own LSP / docset.
  - Only what's installed in the running interpreter's environment is
    visible. Packages not on `sys.path` cannot be queried.

Tools exposed (mirroring the context7 verb shape so callers see the same
mental model):

  - resolve_package(query)
      Find installed packages whose name or summary matches `query`.
      Returns name + version + summary for the top 20 ranked matches.

  - get_docs(target, depth="summary")
      Render docs for a target string that may be a package
      (`requests`), a module (`requests.sessions`), or a fully-qualified
      symbol (`requests.Session.get`). `depth` is "summary" (one-line
      doc + signature) or "full" (pydoc render or full docstring).

  - list_packages(prefix=None, limit=200)
      Enumerate every installed distribution. Useful when the model
      wants to know what's available before guessing names.

Protocol: MCP stdio = newline-delimited JSON-RPC 2.0. We implement the
minimum surface codex actually uses — `initialize`, `tools/list`,
`tools/call` — without pulling in the official `mcp` SDK so tinyctx stays
dependency-light.

CLI:
    python -m tinyctx.pydoc_mcp           # stdio server (what codex spawns)
    python -m tinyctx.pydoc_mcp --selftest # one-shot smoke test, no stdin
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import inspect
import io
import json
import pydoc
import sys
import traceback
from typing import Any

SERVER_NAME = "tinyctx-pydoc"
SERVER_VERSION = "1"
PROTOCOL_VERSION = "2025-06-18"

# Hard cap on payload size returned to the model so a pathological
# request (e.g. get_docs on a huge package) cannot blow up codex's
# context window. ~24KB at 4 chars/token ≈ 6K tokens — enough for a
# substantial module render, small enough not to bury everything else.
MAX_PAYLOAD_CHARS = 24_000


# ─────────────────────────── tool implementations ───────────────────────────


def _truncate(text: str, limit: int = MAX_PAYLOAD_CHARS) -> str:
    if len(text) <= limit:
        return text
    suffix = f"\n\n... [truncated at {limit} chars; full output is {len(text)} chars]"
    return text[: limit - len(suffix)] + suffix


def _iter_distributions() -> list[md.Distribution]:
    """Snapshot of every distribution importlib can see right now."""
    return list(md.distributions())


def _dist_record(dist: md.Distribution) -> dict[str, str]:
    meta = dist.metadata
    return {
        "name": (meta["Name"] or "").strip() or "<unknown>",
        "version": (meta["Version"] or "").strip() or "<unknown>",
        "summary": (meta["Summary"] or "").strip(),
    }


def tool_resolve_package(query: str) -> dict[str, Any]:
    q = (query or "").strip().lower()
    if not q:
        return {"error": "query must be a non-empty string"}
    matches: list[tuple[int, dict[str, str]]] = []
    for dist in _iter_distributions():
        rec = _dist_record(dist)
        name = rec["name"].lower()
        summary = rec["summary"].lower()
        if q == name:
            score = 100
        elif name.startswith(q):
            score = 60
        elif q in name:
            score = 40
        elif q in summary:
            score = 10
        else:
            continue
        matches.append((score, rec))
    matches.sort(key=lambda t: (-t[0], t[1]["name"]))
    top = [rec for _, rec in matches[:20]]
    return {"query": query, "matches": top, "count": len(matches)}


def tool_list_packages(prefix: str | None = None,
                       limit: int = 200) -> dict[str, Any]:
    pkgs: list[dict[str, str]] = []
    for dist in _iter_distributions():
        rec = _dist_record(dist)
        if prefix and not rec["name"].lower().startswith(prefix.lower()):
            continue
        pkgs.append(rec)
    pkgs.sort(key=lambda r: r["name"].lower())
    return {
        "total": len(pkgs),
        "returned": min(limit, len(pkgs)),
        "packages": pkgs[:limit],
    }


def _resolve_module_or_symbol(target: str):
    """Walk a dotted name and return (kind, obj, dotted_name).

    Tries the full path as a module first, then peels suffixes one at a
    time until an importable module is found and the remainder is
    resolved with getattr. Returns ("module"|"symbol"|None, obj|None, str).
    """
    parts = target.split(".")
    for cut in range(len(parts), 0, -1):
        head = ".".join(parts[:cut])
        try:
            mod = importlib.import_module(head)
        except (ImportError, ValueError):
            continue
        except Exception:
            # Why: some packages raise non-Import errors on import
            # (deprecated stubs, side-effect failures). We do not want
            # one bad package to make the whole walk fail.
            continue
        if cut == len(parts):
            return "module", mod, target
        obj: Any = mod
        try:
            for attr in parts[cut:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None, None, target
        return "symbol", obj, target
    return None, None, target


def _render_symbol(obj: Any, dotted: str, *, depth: str) -> str:
    doc = inspect.getdoc(obj) or "(no docstring)"
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = ""
    head = f"{dotted}{sig}\n\n{doc}"
    if depth != "full":
        return head
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        src = ""
    if src:
        return head + "\n\n--- source ---\n" + src
    return head


def _render_module(mod: Any, dotted: str, *, depth: str) -> str:
    if depth == "summary":
        doc = inspect.getdoc(mod) or "(no module docstring)"
        public = sorted(n for n in dir(mod) if not n.startswith("_"))
        return f"module {dotted}\n\n{doc}\n\nexports: {', '.join(public)}"
    # Full render via pydoc — captures the same output as `pydoc requests`.
    try:
        rendered = pydoc.TextDoc().document(mod)
    except Exception as exc:  # pragma: no cover - defensive
        return f"module {dotted}\n\n(pydoc render failed: {exc})"
    return pydoc.plain(rendered)


def tool_get_docs(target: str, depth: str = "summary") -> dict[str, Any]:
    t = (target or "").strip()
    if not t:
        return {"error": "target must be a non-empty string"}
    if depth not in ("summary", "full"):
        return {"error": "depth must be 'summary' or 'full'"}

    kind, obj, dotted = _resolve_module_or_symbol(t)
    if kind is None:
        return {
            "target": t,
            "error": (
                f"could not resolve '{t}'. Tried as module path and "
                "fully-qualified symbol. The package may not be installed "
                "in this environment — use list_packages or "
                "resolve_package to discover what's available."
            ),
        }
    if kind == "module":
        body = _render_module(obj, dotted, depth=depth)
    else:
        body = _render_symbol(obj, dotted, depth=depth)
    return {
        "target": dotted,
        "kind": kind,
        "depth": depth,
        "docs": _truncate(body),
    }


# ─────────────────────────── MCP protocol ───────────────────────────


TOOLS = [
    {
        "name": "resolve_package",
        "description": (
            "Find installed Python packages by name or summary. Returns "
            "ranked matches with name + version + one-line summary. Use "
            "this when you have a fuzzy library name and need to confirm "
            "what's actually available locally."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "name or substring (e.g. 'http', 'fastapi')",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_docs",
        "description": (
            "Render docs for a Python package, module, or symbol from the "
            "local interpreter. `target` may be a package ('requests'), "
            "a module ('requests.sessions'), or a fully-qualified symbol "
            "('requests.Session.get'). `depth='summary'` returns a one-line "
            "doc + signature; `depth='full'` returns pydoc output or source."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "dotted name to look up",
                },
                "depth": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "default": "summary",
                    "description": "summary = short head; full = pydoc/source",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "list_packages",
        "description": (
            "Enumerate every distribution importlib can see in this "
            "interpreter. Filter with `prefix` and cap with `limit`."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prefix": {"type": "string"},
                "limit": {"type": "integer", "default": 200, "minimum": 1},
            },
        },
    },
]


def dispatch_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "resolve_package":
        return tool_resolve_package(args.get("query", ""))
    if name == "get_docs":
        return tool_get_docs(args.get("target", ""),
                              args.get("depth", "summary"))
    if name == "list_packages":
        return tool_list_packages(args.get("prefix"),
                                   int(args.get("limit", 200)))
    return {"error": f"unknown tool: {name}"}


def _ok(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


def handle_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Return a JSON-RPC response dict, or None for notifications."""
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # Notifications (no id) get no response per JSON-RPC.
    is_notification = "id" not in msg

    if method == "initialize":
        return _ok(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method == "notifications/initialized" or method == "initialized":
        return None

    if method == "tools/list":
        return _ok(msg_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments") or {}
        try:
            payload = dispatch_tool(name, args)
        except Exception as exc:
            tb = traceback.format_exc(limit=4)
            payload = {"error": f"{type(exc).__name__}: {exc}",
                       "traceback": tb}
        # MCP content block: text payload as compact JSON so the model
        # can parse it programmatically when useful.
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return _ok(msg_id, {
            "content": [{"type": "text", "text": _truncate(text)}],
            "isError": "error" in payload,
        })

    if method == "ping":
        return _ok(msg_id, {})

    if method == "shutdown":
        return _ok(msg_id, None)

    if is_notification:
        return None
    return _err(msg_id, -32601, f"method not found: {method}")


# ─────────────────────────── stdio loop ───────────────────────────


def serve_stdio(stdin: io.TextIOBase | None = None,
                stdout: io.TextIOBase | None = None) -> None:
    """Read newline-delimited JSON-RPC from stdin, write to stdout."""
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    for line in sin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            err = _err(None, -32700, f"parse error: {exc}")
            sout.write(json.dumps(err) + "\n")
            sout.flush()
            continue
        try:
            resp = handle_message(msg)
        except Exception as exc:
            resp = _err(msg.get("id"), -32603, f"internal error: {exc}")
        if resp is not None:
            sout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sout.flush()


def _selftest() -> int:
    """One-shot smoke test that doesn't touch stdio."""
    r = tool_resolve_package("fastapi")
    assert "matches" in r, r
    r = tool_get_docs("json", depth="summary")
    assert r.get("kind") == "module", r
    r = tool_get_docs("json.dumps", depth="summary")
    assert r.get("kind") == "symbol", r
    r = tool_list_packages(limit=5)
    assert isinstance(r.get("packages"), list), r
    # Protocol-level smoke
    resp = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp and "tools" in resp["result"], resp
    print("pydoc_mcp selftest: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.pydoc_mcp")
    p.add_argument("--selftest", action="store_true",
                   help="run a smoke test and exit (no stdio loop)")
    args = p.parse_args(argv)
    if args.selftest:
        return _selftest()
    # Line-buffered stdout so codex sees each response without an extra flush.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
