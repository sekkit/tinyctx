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


def test_huge_history_does_NOT_auto_escalate_by_default():
    """Aligned with Anthropic Advisor Strategy
    (claude.com/blog/the-advisor-strategy): the EXECUTOR MODEL decides
    when to escalate, infrastructure does not auto-escalate by byte
    count. Default config has all size/turn thresholds disabled.

    A huge body stays on local; the model invokes spawn_agent(advisor)
    when it actually needs strategic input."""
    big = "x" * 4_000_000  # ~1.1M est tokens — way past any old threshold
    body = {"model": "gpt-5.5",
            "input": [{"role": "user", "content": big}]}
    d = decide(body, CFG)
    assert d.route == "local", (
        f"default routing must NOT auto-escalate by size; got {d.route}: {d.reason}"
    )


def test_size_escalation_can_be_re_enabled_for_small_local_backends():
    """Users with a 32k-context LMStudio backend can opt back in by
    setting context_safe_fraction > 0 in config."""
    cfg = Config()
    # Simulate small local backend with size-based escalation explicitly enabled
    cfg.local.context_window = 32_000
    cfg.local.context_safe_fraction = 0.85  # opt-in
    big = "x" * 200_000  # ~55k est tokens, well past 32k×0.85=27k
    body = {"model": "gpt-5.5",
            "input": [{"role": "user", "content": big}]}
    d = decide(body, cfg)
    assert d.route == "frontier", (
        f"explicitly enabled size escalation should fire; got {d.route}: {d.reason}"
    )


def test_turn_count_escalation_can_be_re_enabled():
    """Same opt-in story for turn count."""
    cfg = Config()
    cfg.escalate_turn_count = 15
    body = {
        "input": [
            {"role": "user", "content": f"turn {i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"reply {i}"}
            for i in range(40)
        ]
    }
    d = decide(body, cfg)
    assert d.route == "frontier"
    assert "turn_count" in d.reason


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


def test_context_window_drives_escalation_when_set():
    """When cfg.local.context_window is set, the router uses it (× safe
    fraction) instead of the legacy absolute escalate_input_tokens."""
    from tinyctx.config import Config, BackendCfg
    cfg = Config()
    cfg.local = BackendCfg(
        base_url="x", model="m", wire_api="chat",
        context_window=1_000_000, context_safe_fraction=0.85,
    )
    # 100K input → well below 850K cap → local
    body = {"input": [{"role": "user",
                       "content": [{"type": "input_text", "text": "x" * 360_000}]}]}
    d = decide(body, cfg)
    assert d.route == "local", d.reason

    # 900K input → above 850K cap → frontier
    body = {"input": [{"role": "user",
                       "content": [{"type": "input_text", "text": "x" * 3_500_000}]}]}
    d = decide(body, cfg)
    assert d.route == "frontier", d.reason
    assert "of local ctx 1000000" in d.reason


def test_legacy_threshold_used_when_context_window_unset():
    """If context_window=0, fall back to absolute escalate_input_tokens."""
    from tinyctx.config import Config, BackendCfg
    cfg = Config()
    cfg.local = BackendCfg(base_url="x", model="m", wire_api="chat",
                           context_window=0)
    cfg.escalate_input_tokens = 60_000
    body = {"input": [{"role": "user",
                       "content": [{"type": "input_text", "text": "x" * 250_000}]}]}
    d = decide(body, cfg)
    assert d.route == "frontier"
    assert "60000" in d.reason


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