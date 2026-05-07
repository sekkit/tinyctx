"""Unit-test the router decisions, the compaction fingerprint, and the
encrypted_content sanitizer. No network required."""
from __future__ import annotations

from tinyctx.config import Config
from tinyctx.router import decide, is_compaction_request
from tinyctx.sanitize import strip_encrypted_content


CFG = Config()


def test_compaction_fingerprint_redirects_to_local():
    body = {
        "model": "gpt-5.5",
        "instructions": (
            "Create a handoff summary for another LLM that will resume the "
            "task. Be concise and structured."
        ),
        "input": [{"role": "user", "content": "..."}],
    }
    d = decide(body, CFG)
    assert d.is_compaction is True
    assert d.route == "local", d.reason


def test_short_query_stays_local():
    body = {"model": "gpt-5.5",
            "input": [{"role": "user", "content": [{"type": "input_text",
                       "text": "rename foo() to bar()"}]}]}
    d = decide(body, CFG)
    assert d.route == "local", d.reason
    assert d.is_compaction is False


def test_huge_history_escalates():
    big = "x" * 400_000  # ~111k est tokens
    body = {"model": "gpt-5.5",
            "input": [{"role": "user", "content": big}]}
    d = decide(body, CFG)
    assert d.route == "frontier", d.reason


def test_force_route_overrides_everything():
    cfg = Config()
    cfg.force_route = "frontier"
    body = {"input": [{"role": "user", "content": "tiny"}]}
    d = decide(body, cfg)
    assert d.route == "frontier"

    cfg.force_route = "local"
    body = {"input": [{"role": "user", "content": "x" * 1_000_000}]}
    d = decide(body, cfg)
    assert d.route == "local"


def test_strip_encrypted_content_removes_from_reasoning():
    body = {
        "input": [
            {"role": "user", "content": "hello"},
            {"type": "reasoning", "encrypted_content": "OPAQUE"},
            {"role": "assistant", "content": [
                {"type": "reasoning", "encrypted_content": "ALSO_OPAQUE"},
                {"type": "output_text", "text": "world"},
            ]},
        ],
        "include": ["reasoning.encrypted_content", "tool_calls"],
    }
    out = strip_encrypted_content(body)
    assert "encrypted_content" not in out["input"][1]
    nested = out["input"][2]["content"]
    assert all("encrypted_content" not in c for c in nested if isinstance(c, dict))
    assert "reasoning.encrypted_content" not in out["include"]
    # original unchanged (we deep-copy)
    assert body["input"][1]["encrypted_content"] == "OPAQUE"


def test_compaction_phrase_variants():
    # These are the three invariant phrases from codex's handoff prompt.
    assert is_compaction_request("Create a handoff summary for another LLM")
    assert is_compaction_request("for another LLM that will resume the task next")
    assert is_compaction_request("help the next LLM seamlessly continue the work")
    assert not is_compaction_request("rename foo to bar")
    assert not is_compaction_request("write a handoff document for the team")  # no fingerprint


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
