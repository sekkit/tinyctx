"""Multi-project isolation: per-conversation state must scope to per-project too.

Background: codex.app currently does NOT send `x-codex-session-id`, so
99%+ of real traces have session_id="global". If per-session state
(proactive_compact cache, error_streak, mutation-gate timing) was keyed
on session_id alone, every project working through tinyctx would share
one bucket of state — project A's summary injected into project B's
requests, A's failure escalating B to frontier, etc.

Fix: `proxy._project_session_key(request, sid)` combines sid with a
hash of `x-codex-cwd` (which codex DOES send and auto_scout already
uses) so the composite key is unique per (project, session).

These tests verify:
  - composite key generation (cwd hash + sid)
  - fallback to plain sid when cwd header is absent
  - same sid + different cwd → DIFFERENT keys (the whole point)
  - proactive_compact cache doesn't leak between projects
"""
from __future__ import annotations

import hashlib

import pytest


# ─── _project_session_key helper ───────────────────────────────────────────


class _FakeRequest:
    """Minimal stand-in for FastAPI Request — only needs `.headers.get()`."""
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = _FakeHeaders(headers or {})


class _FakeHeaders:
    def __init__(self, d: dict[str, str]):
        self._d = d

    def get(self, k: str, default=None):
        return self._d.get(k, default)


def test_composite_key_combines_cwd_hash_with_sid():
    from tinyctx.proxy import _project_session_key
    req = _FakeRequest({"x-codex-cwd": "/Users/me/dev/projectA"})
    out = _project_session_key(req, "global")
    # Should be "<8-hex-chars>:global"
    parts = out.split(":")
    assert len(parts) == 2
    assert len(parts[0]) == 8
    int(parts[0], 16)  # must be valid hex (will raise if not)
    assert parts[1] == "global"


def test_same_sid_different_cwd_yields_different_keys():
    """The whole point: two projects with sid='global' must NOT collide."""
    from tinyctx.proxy import _project_session_key
    req_a = _FakeRequest({"x-codex-cwd": "/Users/me/dev/projectA"})
    req_b = _FakeRequest({"x-codex-cwd": "/Users/me/dev/projectB"})
    key_a = _project_session_key(req_a, "global")
    key_b = _project_session_key(req_b, "global")
    assert key_a != key_b, (
        f"different projects with same sid must produce different keys; "
        f"got {key_a!r} == {key_b!r}"
    )


def test_falls_back_to_plain_sid_when_cwd_missing():
    """If codex didn't send x-codex-cwd (older versions, or non-codex
    clients), behave as before — no regression."""
    from tinyctx.proxy import _project_session_key
    req = _FakeRequest({})
    out = _project_session_key(req, "global")
    assert out == "global"


def test_falls_back_to_plain_sid_when_cwd_empty():
    from tinyctx.proxy import _project_session_key
    req = _FakeRequest({"x-codex-cwd": ""})
    out = _project_session_key(req, "global")
    assert out == "global"


def test_same_cwd_same_sid_is_stable_across_calls():
    """Determinism: a given (cwd, sid) pair always produces the same key.
    Otherwise cache hits would never happen."""
    from tinyctx.proxy import _project_session_key
    req = _FakeRequest({"x-codex-cwd": "/Users/me/dev/foo"})
    a = _project_session_key(req, "global")
    b = _project_session_key(req, "global")
    assert a == b


def test_cwd_path_is_hashed_not_emitted_raw():
    """The composite key shouldn't contain the raw cwd path (filesystem
    paths can be sensitive — usernames, repo names — and the keys flow
    into trace JSONL logs which are user-readable)."""
    from tinyctx.proxy import _project_session_key
    req = _FakeRequest({"x-codex-cwd": "/Users/secret-username/private-repo"})
    out = _project_session_key(req, "global")
    assert "secret-username" not in out
    assert "private-repo" not in out
    assert "/Users" not in out


# ─── proactive_compact cross-project isolation ─────────────────────────────


def test_proactive_compact_cache_isolates_by_composite_key():
    """If two callers use DIFFERENT composite keys, their summaries must
    NOT collide. Same input items but different keys → different cache
    entries (= different summaries)."""
    from tinyctx.sanitize import (
        proactive_compact, clear_proactive_cache, _PROACTIVE_SUMMARY_CACHE,
    )
    clear_proactive_cache()

    summarizer_calls: list[str] = []

    def fake_summarizer(blob: str) -> str:
        summarizer_calls.append(blob[:30])
        return f"summary-#{len(summarizer_calls)}"

    body = {
        "model": "x", "instructions": "y",
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": f"item-{i}"}]}
                  for i in range(40)],
    }
    # Two projects with the same items would produce the same summary
    # under the OLD design (key = session_id only) because they'd share
    # ('global', bucket). With composite key, they get isolated entries.
    out_a, info_a = proactive_compact(
        body, session_id="cwdA:global", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=fake_summarizer)
    out_b, info_b = proactive_compact(
        body, session_id="cwdB:global", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=fake_summarizer)
    # both must apply
    assert info_a["applied"] and info_b["applied"]
    # both must MISS cache (different composite keys)
    assert info_a["cached"] is False
    assert info_b["cached"] is False
    # summarizer was called TWICE (once per project)
    assert len(summarizer_calls) == 2

    # Subsequent call from project A should HIT (uses A's cache, not B's)
    out_a2, info_a2 = proactive_compact(
        body, session_id="cwdA:global", est_tokens=300_000,
        threshold_tokens=200_000, recent_keep=8,
        summarizer=fake_summarizer)
    assert info_a2["cached"] is True
    # No new summarizer call
    assert len(summarizer_calls) == 2


def test_clear_proactive_cache_per_project_does_not_clear_other_projects():
    """clear_proactive_cache(session_id=...) targeted clear must only
    remove entries for that key, not other projects'."""
    from tinyctx.sanitize import (
        proactive_compact, clear_proactive_cache, _PROACTIVE_SUMMARY_CACHE,
    )
    clear_proactive_cache()

    body = {"model":"x","instructions":"y",
            "input":[{"type":"message","role":"user",
                      "content":[{"type":"input_text","text":f"i{i}"}]}
                     for i in range(40)]}
    proactive_compact(body, session_id="A:global", est_tokens=300_000,
                      threshold_tokens=200_000, recent_keep=8,
                      summarizer=lambda b: "A-summary")
    proactive_compact(body, session_id="B:global", est_tokens=300_000,
                      threshold_tokens=200_000, recent_keep=8,
                      summarizer=lambda b: "B-summary")
    a_keys = [k for k in _PROACTIVE_SUMMARY_CACHE if k[0] == "A:global"]
    b_keys = [k for k in _PROACTIVE_SUMMARY_CACHE if k[0] == "B:global"]
    assert a_keys and b_keys

    # Clear only A
    clear_proactive_cache("A:global")
    a_keys_after = [k for k in _PROACTIVE_SUMMARY_CACHE if k[0] == "A:global"]
    b_keys_after = [k for k in _PROACTIVE_SUMMARY_CACHE if k[0] == "B:global"]
    assert not a_keys_after  # A gone
    assert b_keys_after == b_keys  # B intact


# ─── error_streak isolation (via proxy module's global dict) ───────────────


def test_error_streak_isolates_by_composite_key():
    """Two projects accumulating errors under sid='global' must not
    cross-pollute the streak counter. Project A's failures shouldn't
    bump project B's streak past the auto-escalate threshold."""
    from tinyctx import proxy
    # Reset
    proxy._SESSION_ERROR_STREAK.clear()

    proxy._SESSION_ERROR_STREAK["A:global"] += 1
    proxy._SESSION_ERROR_STREAK["A:global"] += 1
    proxy._SESSION_ERROR_STREAK["A:global"] += 1
    # B never errored
    assert proxy._SESSION_ERROR_STREAK["A:global"] == 3
    assert proxy._SESSION_ERROR_STREAK["B:global"] == 0


# ─── trace propagation ─────────────────────────────────────────────────────


def test_trace_carries_project_session_key():
    """The composite key should be visible in trace records so users can
    debug "why is this turn escalating?" / "why did proactive_compact
    miss cache?" — the per-project scoping has to be observable."""
    from tinyctx.trace import RequestTrace
    t = RequestTrace(session_id="global")
    t.project_session_key = "abc12345:global"
    # Should round-trip through dataclass asdict
    from dataclasses import asdict
    d = asdict(t)
    assert d["project_session_key"] == "abc12345:global"


# ─── _conversation_session_key helper ──────────────────────────────────────


def test_conv_key_appends_prompt_cache_key():
    """Codex sends `prompt_cache_key` as a UUID stable per conversation.
    Composite key = proj_sid + ':' + that uuid."""
    from tinyctx.proxy import _conversation_session_key
    body = {"prompt_cache_key": "019e0741-f905-7332-b88a-5414ec53c31d"}
    out = _conversation_session_key("cwdhash:global", body)
    assert out == "cwdhash:global:019e0741-f905-7332-b88a-5414ec53c31d"


def test_conv_key_falls_back_to_proj_sid_when_missing():
    """No prompt_cache_key field → back-compat: just proj_sid."""
    from tinyctx.proxy import _conversation_session_key
    assert _conversation_session_key("global", {}) == "global"
    assert _conversation_session_key("p:global", {"model": "x"}) == "p:global"


def test_conv_key_falls_back_when_empty_string():
    """Empty string prompt_cache_key → fall back."""
    from tinyctx.proxy import _conversation_session_key
    assert _conversation_session_key("p", {"prompt_cache_key": ""}) == "p"


def test_conv_key_distinct_for_different_conversations():
    """Same project, two different prompt_cache_key values → distinct keys.
    This is the property the three watchdog modules rely on."""
    from tinyctx.proxy import _conversation_session_key
    body_a = {"prompt_cache_key": "019e0741-aaaa"}
    body_b = {"prompt_cache_key": "019e14c1-bbbb"}
    key_a = _conversation_session_key("p:global", body_a)
    key_b = _conversation_session_key("p:global", body_b)
    assert key_a != key_b
    # Both start with the same project prefix
    assert key_a.startswith("p:global:")
    assert key_b.startswith("p:global:")


# ─── stable-fingerprint resolution (pck-drift bug fix) ─────────────────────
#
# The new key shape is `proj_sid:fp:<8-hex>` when codex sends any of:
#   - client_metadata.x-codex-installation-id
#   - body.model
#   - body.input[0..2] developer-role text
# That fingerprint is computed from the union of those signals and is
# STABLE across prompt_cache_key drift. When none of those signals are
# present we fall back to `proj_sid:<pck>` and then to bare `proj_sid`.

def _codex_body(*, pck="pck-A", install="inst-1", model="gpt-5.5",
                dev_text="<permissions>danger-full-access</permissions>"):
    """Minimal codex-shaped body for fingerprint tests."""
    body = {"model": model, "prompt_cache_key": pck,
            "client_metadata": {"x-codex-installation-id": install},
            "input": [{"type": "message", "role": "developer",
                       "content": [{"type": "input_text", "text": dev_text}]}]}
    return body


def test_conv_key_stable_across_prompt_cache_key_drift():
    """The original bug: in a long stuck saga the prompt_cache_key flips
    mid-conversation (OpenAI prompt-cache invalidation), which reset the
    synthetic_continue counter on every drift event. The fingerprint
    must produce IDENTICAL keys across pck drift when the structural
    signals are unchanged."""
    from tinyctx.proxy import _conversation_session_key
    body_t1 = _codex_body(pck="019e0741-aaaa")
    body_t2 = _codex_body(pck="019e14c1-bbbb")  # drift!
    body_t3 = _codex_body(pck="019e1700-cccc")  # more drift!
    k1 = _conversation_session_key("proj:global", body_t1)
    k2 = _conversation_session_key("proj:global", body_t2)
    k3 = _conversation_session_key("proj:global", body_t3)
    assert k1 == k2 == k3, f"keys must be stable across pck drift; got {k1!r} {k2!r} {k3!r}"
    assert k1.startswith("proj:global:fp:")


def test_conv_key_distinguishes_advisor_subthread_by_model():
    """Advisor sub-threads use a different model (`tinyctx-frontier`)
    and a different body shape. Even on the same install, they must
    produce a DIFFERENT conv_sid from the main thread to keep their
    counters isolated."""
    from tinyctx.proxy import _conversation_session_key
    main = _codex_body(model="gpt-5.5")
    advisor = _codex_body(model="tinyctx-frontier")
    k_main = _conversation_session_key("proj:global", main)
    k_adv = _conversation_session_key("proj:global", advisor)
    assert k_main != k_adv


def test_conv_key_distinguishes_different_installations():
    """Two codex installs on the same machine (rare but possible) must
    not share counters."""
    from tinyctx.proxy import _conversation_session_key
    b1 = _codex_body(install="inst-A")
    b2 = _codex_body(install="inst-B")
    assert _conversation_session_key("p", b1) != _conversation_session_key("p", b2)


def test_conv_key_distinguishes_sandbox_mode_change():
    """Different developer-block content (different sandbox mode or
    permissions config) → different key. This is the proxy for 'genuinely
    different conversation context'."""
    from tinyctx.proxy import _conversation_session_key
    b1 = _codex_body(dev_text="<permissions>danger-full-access</permissions>")
    b2 = _codex_body(dev_text="<permissions>workspace-write</permissions>")
    assert _conversation_session_key("p", b1) != _conversation_session_key("p", b2)


def test_conv_key_falls_back_to_pck_when_no_stable_signals():
    """If client_metadata, model, and input are all absent (older clients
    or non-codex callers), the resolver gracefully falls back to the
    legacy `proj_sid:pck` form."""
    from tinyctx.proxy import _conversation_session_key
    body = {"prompt_cache_key": "legacy-uuid-1234"}
    out = _conversation_session_key("p:global", body)
    assert out == "p:global:legacy-uuid-1234"


def test_conv_key_falls_back_to_proj_sid_when_nothing_present():
    """Empty body (compaction handoff, etc.) → bare proj_sid."""
    from tinyctx.proxy import _conversation_session_key
    assert _conversation_session_key("p:global", {}) == "p:global"


def test_conv_key_same_within_one_real_conversation():
    """Realistic scenario: same codex install / model / first-dev-block
    across many turns → SAME key. This is the invariant the budget cap
    depends on for monotonic accumulation."""
    from tinyctx.proxy import _conversation_session_key
    keys = set()
    for turn_pck in ["a", "b", "c", "d", "e", "f"]:  # simulate drift each turn
        body = _codex_body(pck=turn_pck)
        keys.add(_conversation_session_key("proj:global", body))
    assert len(keys) == 1, f"expected single stable key across 6 turns, got {keys}"
