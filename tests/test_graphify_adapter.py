"""Tests for the graphify -> tinyctx graph adapter."""
from __future__ import annotations

from tinyctx.graphify_adapter import _approx_tokens, _split_signature_body, adapt


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
