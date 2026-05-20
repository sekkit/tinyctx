"""Tests for the graphify -> tinyctx graph adapter."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from tinyctx import graphify_adapter as ga
from tinyctx.graphify_adapter import (
    _approx_tokens,
    _extract_deps,
    _extract_id,
    _extract_text,
    _split_signature_body,
    adapt,
    main,
)


def test_node_link_form():
    g = {
        "nodes": [
            {"id": "a", "code": "def a():\n    return b()"},
            {"id": "b", "code": "def b():\n    pass"},
        ],
        "links": [
            {"source": "a", "target": "b"},
        ],
    }
    out = adapt(g)
    ids = {n["id"] for n in out["nodes"]}
    assert ids == {"a", "b"}
    a = next(n for n in out["nodes"] if n["id"] == "a")
    assert "b" in a["deps"]
    # signature is the first line; body is the rest
    assert a["wrapped_signature"] >= 1
    assert a["wrapped_body"] >= 1


def test_flat_form_with_inline_deps():
    g = {
        "nodes": [
            {"id": "x", "text": "header\nbody body body", "deps": ["y", "z"]},
            {"id": "y", "text": "y body"},
            {"id": "z", "text": "z body"},
        ]
    }
    out = adapt(g)
    x = next(n for n in out["nodes"] if n["id"] == "x")
    assert set(x["deps"]) == {"y", "z"}


def test_dedup_by_id():
    g = {"nodes": [{"id": "a"}, {"id": "a"}]}
    out = adapt(g)
    assert len(out["nodes"]) == 1


def test_signature_body_split():
    sig, body = _split_signature_body("def foo(x):\n    return x + 1\n")
    assert sig >= 1
    assert body >= 1
    sig2, body2 = _split_signature_body("oneline")
    assert sig2 >= 1
    assert body2 == 0


def test_approx_tokens_handles_empty():
    assert _approx_tokens("") == 0
    assert _approx_tokens(None) == 0
    assert _approx_tokens("a b c") >= 3


def test_skips_unidentifiable_nodes():
    g = {"nodes": [{"id": "a"}, {"text": "no id here"}]}
    out = adapt(g)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["id"] == "a"


# ---------------------------------------------------- new behavioral tests


def test_empty_graph_yields_empty_nodes():
    """adapt({}) and adapt({nodes: []}) both return {'nodes': []} cleanly."""
    assert adapt({}) == {"nodes": []}
    assert adapt({"nodes": []}) == {"nodes": []}


def test_node_link_with_dict_source_and_target():
    """When links carry dict-shaped source/target (NetworkX object form),
    adapt() must still wire deps via the node's id field."""
    g = {
        "nodes": [
            {"id": "a", "code": "def a():\n    call b"},
            {"id": "b", "code": "def b(): pass"},
        ],
        "links": [
            {"source": {"id": "a"}, "target": {"id": "b"}},
        ],
    }
    out = adapt(g)
    a = next(n for n in out["nodes"] if n["id"] == "a")
    assert "b" in a["deps"]


def test_node_link_with_edges_alias():
    """Some emitters write the edge list under `edges` instead of `links`.
    adapt() should accept both."""
    g = {
        "nodes": [{"id": "a", "code": "X"}, {"id": "b", "code": "Y"}],
        "edges": [{"source": "a", "target": "b"}],
    }
    out = adapt(g)
    a = next(n for n in out["nodes"] if n["id"] == "a")
    assert a["deps"] == ["b"]


def test_extract_id_prefers_explicit_keys_in_order():
    """`_extract_id` falls through id → name → qualified_name → fullname → path."""
    assert _extract_id({"id": "x", "name": "y"}) == "x"
    assert _extract_id({"name": "y", "qualified_name": "z"}) == "y"
    assert _extract_id({"qualified_name": "z", "path": "p"}) == "z"
    assert _extract_id({"path": "p"}) == "p"
    assert _extract_id({"unrelated": "u"}) is None
    # Non-string id values are ignored
    assert _extract_id({"id": 42}) is None
    # Empty string is treated as missing
    assert _extract_id({"id": ""}) is None


def test_extract_text_falls_through_keys():
    """`_extract_text` returns the first available text-bearing field."""
    assert _extract_text({"code": "src"}) == "src"
    assert _extract_text({"snippet": "snip"}) == "snip"
    # When nothing matches, returns empty string (not None).
    assert _extract_text({}) == ""
    # Non-string values are ignored.
    assert _extract_text({"code": 5, "label": "fallback"}) == "fallback"


def test_extract_deps_handles_string_and_dict_entries():
    """`_extract_deps` accepts both bare strings and dict entries with
    nested ids; it concatenates across keys."""
    node = {
        "deps": ["a", "b", {"id": "c"}],
        "calls": [{"id": "d"}, "e"],
        "imports": "not-a-list",  # ignored
    }
    deps = _extract_deps(node)
    # Order: first deps then calls
    assert deps == ["a", "b", "c", "d", "e"]


def test_dedup_preserves_first_occurrence_data():
    """When two nodes share an id, the first occurrence wins (its
    signature/body/deps are retained)."""
    g = {"nodes": [
        {"id": "a", "code": "first\nfirst body", "deps": ["x"]},
        {"id": "a", "code": "second\nsecond body different", "deps": ["y"]},
    ]}
    out = adapt(g)
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["deps"] == ["x"]


def test_approx_tokens_punctuation_split():
    """`_approx_tokens` splits around non-word characters too (slight
    overestimate, but stable)."""
    n = _approx_tokens("foo(bar)")
    assert n >= 4   # foo, (, bar, )
    # Single token always returns at least 1 (per `max(1, ...)` floor).
    assert _approx_tokens("solo") == 1
    # Only empty/None inputs return 0 — every other non-empty string
    # gets the floor of 1.
    assert _approx_tokens("   ") == 1


def test_split_signature_body_only_signature_when_no_newline():
    """Single-line input has signature tokens > 0 and body == 0."""
    sig, body = _split_signature_body("single_line_only")
    assert sig >= 1
    assert body == 0
    # Trailing-newline input still treats post-newline part (empty) as 0.
    sig2, body2 = _split_signature_body("def x():\n")
    assert sig2 >= 1
    assert body2 == 0


def test_main_help_flags_return_2():
    """`-h` and `--help` print usage and return rc=2 (no convert)."""
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc1 = main(["-h"])
    assert rc1 == 2
    assert "usage:" in buf.getvalue()
    buf2 = io.StringIO()
    with redirect_stderr(buf2):
        rc2 = main(["--help"])
    assert rc2 == 2


def test_main_no_args_returns_2():
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc = main([])
    assert rc == 2
    assert "usage:" in buf.getvalue()


def test_main_writes_to_stdout_without_out_flag(tmp_path):
    """When --out is omitted, the converted graph goes to stdout as JSON."""
    src = tmp_path / "g.json"
    src.write_text(json.dumps({
        "nodes": [{"id": "n", "code": "def n(): pass"}],
        "links": [],
    }))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([str(src)])
    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert parsed["nodes"][0]["id"] == "n"


def test_main_writes_out_file(tmp_path):
    """--out writes the converted graph to the specified path."""
    src = tmp_path / "g.json"
    src.write_text(json.dumps({"nodes": [{"id": "z", "code": "z"}]}))
    out = tmp_path / "converted.json"
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc = main([str(src), "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    parsed = json.loads(out.read_text())
    assert {n["id"] for n in parsed["nodes"]} == {"z"}
    assert "wrote 1 nodes" in buf.getvalue()


def test_main_malformed_json_raises():
    """Malformed input JSON should surface JSONDecodeError to the caller
    (no silent fallback)."""
    p = Path("/tmp/__tinyctx_does_not_exist_for_sure__.json")
    # Use a file we know is not JSON
    p.write_text("{not json")
    try:
        with pytest.raises(json.JSONDecodeError):
            main([str(p)])
    finally:
        p.unlink(missing_ok=True)


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
    sys.exit(failed)
