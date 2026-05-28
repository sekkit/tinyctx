"""Tests for the in-tree pydoc-mcp server."""
from __future__ import annotations

import io
import json

import pytest

from tinyctx import pydoc_mcp as pm


# ─────────────────────────── tool implementations ───────────────────────────


def test_resolve_package_rejects_empty():
    r = pm.tool_resolve_package("")
    assert "error" in r


def test_resolve_package_finds_pytest_by_exact_name():
    r = pm.tool_resolve_package("pytest")
    assert r["count"] >= 1
    names = [m["name"] for m in r["matches"]]
    assert any(n.lower() == "pytest" for n in names)
    # Exact match must be ranked above substring matches.
    assert r["matches"][0]["name"].lower() == "pytest"


def test_resolve_package_substring_works():
    r = pm.tool_resolve_package("test")
    # 'pytest' contains 'test' — should appear in matches even if not exact.
    names = [m["name"].lower() for m in r["matches"]]
    assert any("test" in n for n in names)


def test_resolve_package_empty_match_returns_zero():
    r = pm.tool_resolve_package("definitelynotapackagexyz123")
    assert r["count"] == 0
    assert r["matches"] == []


def test_list_packages_prefix_filter():
    r = pm.tool_list_packages(prefix="py", limit=50)
    assert r["total"] >= 1
    for pkg in r["packages"]:
        assert pkg["name"].lower().startswith("py")


def test_list_packages_respects_limit():
    r = pm.tool_list_packages(limit=3)
    assert r["returned"] <= 3
    assert len(r["packages"]) <= 3


def test_get_docs_summary_on_stdlib_module():
    r = pm.tool_get_docs("json", depth="summary")
    assert r["kind"] == "module"
    assert "exports" in r["docs"]
    assert "dumps" in r["docs"]  # json.dumps is a well-known export


def test_get_docs_summary_on_stdlib_symbol():
    r = pm.tool_get_docs("json.dumps", depth="summary")
    assert r["kind"] == "symbol"
    # Signature renders something with parens.
    assert "(" in r["docs"] and ")" in r["docs"]
    # Docstring mentions return value / serialization.
    assert "JSON" in r["docs"] or "string" in r["docs"]


def test_get_docs_full_includes_source_for_python_symbol():
    # Use a small pure-Python stdlib function so getsource works reliably.
    r = pm.tool_get_docs("textwrap.indent", depth="full")
    assert r["kind"] == "symbol"
    # Either source slice or pydoc render — both contain 'def'.
    assert "def " in r["docs"]


def test_get_docs_unknown_target_returns_error():
    r = pm.tool_get_docs("not_a_real_module_xyz_123")
    assert "error" in r
    assert "could not resolve" in r["error"]


def test_get_docs_rejects_bad_depth():
    r = pm.tool_get_docs("json", depth="medium")
    assert "error" in r


def test_get_docs_truncates_huge_output():
    """Pathological deep pydoc render must not exceed the cap."""
    r = pm.tool_get_docs("xml.etree.ElementTree", depth="full")
    # 'docs' is the rendered body; we cap at MAX_PAYLOAD_CHARS.
    assert len(r["docs"]) <= pm.MAX_PAYLOAD_CHARS


# ─────────────────────────── MCP protocol layer ─────────────────────────────


def test_handle_initialize():
    resp = pm.handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
    })
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == pm.SERVER_NAME
    assert "tools" in resp["result"]["capabilities"]


def test_handle_initialized_notification_returns_none():
    resp = pm.handle_message({
        "jsonrpc": "2.0", "method": "notifications/initialized",
    })
    assert resp is None


def test_handle_tools_list_returns_three_tools():
    resp = pm.handle_message({
        "jsonrpc": "2.0", "id": 2, "method": "tools/list",
    })
    tools = resp["result"]["tools"]
    names = [t["name"] for t in tools]
    assert names == ["resolve_package", "get_docs", "list_packages"]
    for t in tools:
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"


def test_handle_tools_call_resolve_package():
    resp = pm.handle_message({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "resolve_package",
                   "arguments": {"query": "pytest"}},
    })
    content = resp["result"]["content"]
    assert content[0]["type"] == "text"
    payload = json.loads(content[0]["text"])
    assert payload["count"] >= 1
    assert resp["result"]["isError"] is False


def test_handle_tools_call_error_flag_set_on_error_payload():
    resp = pm.handle_message({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "get_docs",
                   "arguments": {"target": "definitely_missing_xyz"}},
    })
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert "error" in payload
    assert resp["result"]["isError"] is True


def test_handle_unknown_method_returns_method_not_found():
    resp = pm.handle_message({
        "jsonrpc": "2.0", "id": 5, "method": "totally/unknown",
    })
    assert resp["error"]["code"] == -32601


def test_dispatch_unknown_tool_returns_error_payload():
    r = pm.dispatch_tool("not_a_tool", {})
    assert "error" in r


# ─────────────────────────── stdio loop ─────────────────────────────


def test_serve_stdio_handles_initialize_then_list():
    """Drive the full stdio loop with two newline-delimited messages
    and confirm we get back two JSON-RPC responses on stdout."""
    msgs = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]
    stdin = io.StringIO("\n".join(msgs) + "\n")
    stdout = io.StringIO()
    pm.serve_stdio(stdin=stdin, stdout=stdout)
    lines = [ln for ln in stdout.getvalue().split("\n") if ln.strip()]
    assert len(lines) == 2
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["id"] == 1 and "serverInfo" in r1["result"]
    assert r2["id"] == 2 and "tools" in r2["result"]


def test_serve_stdio_parse_error_returns_error_response():
    stdin = io.StringIO("not json at all\n")
    stdout = io.StringIO()
    pm.serve_stdio(stdin=stdin, stdout=stdout)
    resp = json.loads(stdout.getvalue().strip())
    assert resp["error"]["code"] == -32700


def test_serve_stdio_skips_blank_lines():
    stdin = io.StringIO("\n\n\n")
    stdout = io.StringIO()
    pm.serve_stdio(stdin=stdin, stdout=stdout)
    assert stdout.getvalue() == ""


# ─────────────────────────── selftest ─────────────────────────────


def test_main_selftest_exit_zero(capsys):
    rc = pm.main(["--selftest"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ok" in out
