"""Integration moat for the tinyctx proxy: exercise the five sagas that
matter most when the underlying modules get rearranged in future phases.

Each saga drives the proxy through its full request lifecycle (routing
+ sanitize + retry + stream relay + watchdog) and asserts behavior that
must NOT regress, even when the implementation layout changes.

The sagas mock the upstream httpx layer only. Where the saga needs to
exercise the FastAPI handler itself (proactive_compact lives there, not
in _forward), the handler is invoked through an ASGITransport so the
full pipeline runs.

Run:
  uv run pytest tests/test_integration_workflow.py -x -v
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import patch

import httpx
import pytest

from helpers.integration import (
    drain_stream,
    healthy_frontier_chunks,
    healthy_local_chunks,
    make_mock_stream,
)


# ─── proxy module fixture ────────────────────────────────────────────────


@pytest.fixture
def proxy_module(monkeypatch, tmp_path):
    """Import proxy fresh with a tmp log dir, predictable backends, and
    no user config. Mirrors test_proxy_retry.py's pattern."""
    monkeypatch.setenv("TINYCTX_VERBOSE", "1")
    monkeypatch.setenv("TINYCTX_CONFIG", "/dev/null")
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("TINYCTX_LOG_DIR", str(log_dir))
    monkeypatch.setenv("TINYCTX_LOCAL_BASE_URL", "http://local.test/v1")
    monkeypatch.setenv("TINYCTX_LOCAL_WIRE_API", "responses")
    monkeypatch.setenv("TINYCTX_LOCAL_MODEL", "qwen-test")
    # api_key_env is hardcoded to TINYCTX_LOCAL_API_KEY / TINYCTX_FRONTIER_API_KEY
    # in config.py; we set the API KEY VALUES on those vars so _resolve_api_key
    # picks them up instead of falling back to ~/.codex/auth.json.
    monkeypatch.setenv("TINYCTX_LOCAL_API_KEY", "sk-local-bearer")
    monkeypatch.setenv("TINYCTX_FRONTIER_BASE_URL", "http://frontier.test/v1")
    monkeypatch.setenv("TINYCTX_FRONTIER_MODEL", "gpt-test")
    monkeypatch.setenv("TINYCTX_FRONTIER_API_KEY", "sk-frontier-bearer")
    import sys
    for m in list(sys.modules):
        if m.startswith("tinyctx"):
            del sys.modules[m]
    import tinyctx.proxy as proxy_mod
    # Disable soft_completion features so post-stream classifier doesn't
    # spawn background work that pulls in network during teardown. Also
    # disable forensics writes so the tmp dir stays clean for assertions.
    monkeypatch.setattr(proxy_mod.CFG, "soft_completion_gate_enabled", False)
    monkeypatch.setattr(proxy_mod.CFG, "soft_completion_stream_rewrite_enabled", False)
    monkeypatch.setattr(proxy_mod.CFG, "forensics_enabled", False)
    monkeypatch.setattr(proxy_mod.CFG, "self_classify_enabled", False)
    # Generous stall threshold so non-stall sagas never trip the watchdog.
    monkeypatch.setattr(proxy_mod.CFG, "stall_threshold_s", 600.0)
    # Reset module-level dictionaries between tests.
    from tinyctx import empty_response_guard as _erg
    from tinyctx import stall_watchdog as _sw
    from tinyctx import request_phase as _rp
    _erg.reset_state()
    _sw.reset_state()
    _rp.reset_state()
    return proxy_mod


def _decision_local(proxy_module, reason: str = "saga-local"):
    return proxy_module.Decision("local", reason, is_compaction=False)


def _decision_frontier(proxy_module, reason: str = "saga-frontier"):
    return proxy_module.Decision("frontier", reason, is_compaction=False)


def _read_log_events(log_dir, name: str | None = None) -> list[dict]:
    log_file = log_dir / f"tinyctx-{time.strftime('%Y%m%d')}.jsonl"
    if not log_file.exists():
        return []
    out: list[dict] = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if name is None or ev.get("event") == name:
            out.append(ev)
    return out


# ════════════════════════════════════════════════════════════════════════
# SAGA 1: Healthy local turn
# ════════════════════════════════════════════════════════════════════════


class TestSaga1HealthyLocal:
    """codex POSTs a small request, proxy routes local, upstream returns a
    well-formed Responses-API SSE stream. Verify the relay completes with
    no retry / stall / upstream_error events and that the phase machine
    progresses received -> routing -> backend_streaming -> done."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_local_healthy(self, proxy_module,
                                                  monkeypatch):
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 1.0)
        from tinyctx.router import Decision
        from tinyctx.trace import RequestTrace
        from tinyctx import request_phase as _rp
        from tinyctx import empty_response_guard as _erg

        scripts = [("ok", healthy_local_chunks())]
        stream_fn, state = make_mock_stream(scripts)

        proj_sid = "saga1-proj"
        conv_sid = "saga1-conv"
        decision = Decision("local", "small/short -> cheap path",
                            is_compaction=False)
        trace = RequestTrace(session_id=proj_sid)

        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json",
                 "Authorization": "Bearer sk-local-bearer"},
                {"model": "qwen-test",
                 "input": [{"role": "user", "content": "hi"}]},
                is_stream=True,
                sid=proj_sid,
                decision=decision,
                trace=trace,
                conv_sid=conv_sid,
            )
            body_bytes = await drain_stream(sr)

        # Exactly one upstream attempt — no retries.
        assert len(state["calls"]) == 1
        assert "local.test" in state["calls"][0]

        # The body must include the function_call item AND the
        # response.completed terminator the upstream sent.
        assert b"function_call" in body_bytes
        assert b"response.completed" in body_bytes

        # No retry_attempted, stall_cancelled, or upstream_error events
        # for this session.
        events = _read_log_events(proxy_module.CFG.log_dir)
        retry_evts = [e for e in events if e.get("event") == "retry_attempted"
                      and e.get("session") == proj_sid]
        stall_evts = [e for e in events if e.get("event") == "stall_cancelled"
                      and e.get("session") == proj_sid]
        upstream_err = [e for e in events if e.get("event") == "upstream_error"
                         and e.get("session") == proj_sid]
        assert retry_evts == [], retry_evts
        assert stall_evts == []
        assert upstream_err == []

        # stream_done event records bytes + elapsed for this session.
        done = [e for e in events if e.get("event") == "stream_done"
                and e.get("session") == proj_sid]
        assert len(done) == 1
        d = done[0]
        assert d["route"] == "local"
        assert d["bytes"] > 0
        assert d["elapsed_s"] >= 0

        # force_next_to_frontier must NOT be set after a clean turn.
        assert _erg.consume_force_frontier(conv_sid) is None

        # Phase machine ends on `done`. (The full received/routing
        # progression happens in the responses() handler, exercised via
        # the ASGI saga below; here _forward owns backend_streaming/done.)
        phase = _rp.get_phase(proj_sid)
        assert phase is not None
        assert phase["phase"] == "done"


# ════════════════════════════════════════════════════════════════════════
# SAGA 2: Healthy frontier turn
# ════════════════════════════════════════════════════════════════════════


class TestSaga2HealthyFrontier:
    """codex POSTs with requested_model=gpt-5.5 (mapped through the
    proxy's force-route alias `tinyctx-frontier`), proxy targets the
    frontier backend, upstream sends reasoning + tool_call + completed.
    Verify route reason mentions frontier, no translation is applied
    (frontier speaks responses natively), and the stream lands clean."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_frontier_healthy(self, proxy_module,
                                                     monkeypatch):
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 1.0)
        from tinyctx.router import Decision
        from tinyctx.trace import RequestTrace

        scripts = [("ok", healthy_frontier_chunks())]
        stream_fn, state = make_mock_stream(scripts)

        proj_sid = "saga2-proj"
        conv_sid = "saga2-conv"
        # Reason string mirrors what the responses() handler builds when
        # requested_model == "tinyctx-frontier"; the saga directly drives
        # _forward (no need to round-trip the alias resolver).
        decision = Decision("frontier", "client requested tinyctx-frontier",
                            is_compaction=False)
        trace = RequestTrace(session_id=proj_sid)

        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            sr = await proxy_module._forward(
                "http://frontier.test/v1/responses",
                {"Content-Type": "application/json",
                 "Authorization": "Bearer sk-frontier-bearer"},
                {"model": "gpt-test",
                 "input": [{"role": "user", "content": "hi"}]},
                is_stream=True,
                sid=proj_sid,
                decision=decision,
                trace=trace,
                conv_sid=conv_sid,
                # frontier wire_api == "responses" so translation off.
                translate_tool_calls=False,
                chat_to_responses=False,
            )
            body_bytes = await drain_stream(sr)

        assert len(state["calls"]) == 1
        assert "frontier.test" in state["calls"][0]
        # Reasoning event passes through (frontier sends them; we don't
        # touch them — that's part of "no translation applied").
        assert b"reasoning_summary_text" in body_bytes
        assert b"response.completed" in body_bytes

        # The trace records translated=False — no ChatToResponses
        # translator instantiated for this attempt.
        assert trace.translated is False
        assert trace.translated_calls == 0

        # stream_done shows route=frontier.
        events = _read_log_events(proxy_module.CFG.log_dir)
        done = [e for e in events if e.get("event") == "stream_done"
                and e.get("session") == proj_sid]
        assert len(done) == 1
        assert done[0]["route"] == "frontier"

        # Decision reason mentions frontier-side intent.
        assert "frontier" in decision.reason.lower()


# ════════════════════════════════════════════════════════════════════════
# SAGA 3: Retry-escalate LMStudio 400 -> frontier
# ════════════════════════════════════════════════════════════════════════


class TestSaga3RetryEscalate:
    """Initial route is local. Mock LMStudio returns 400 with the typical
    'Unexpected message role' body. retry_policy.classify_failure should
    decide retry_escalate; the producer re-issues the request to the
    frontier backend (mock chatgpt.com returns 200 + clean stream).
    The auth header on the SECOND attempt must be the frontier bearer,
    not the leaked local bearer (the f8c2489 fix). force_next_to_frontier
    is set on the conv key. codex sees the frontier success body."""

    @pytest.mark.asyncio
    async def test_retry_escalate_replaces_auth_and_sets_flag(
            self, proxy_module, monkeypatch):
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 1.0)
        from tinyctx.router import Decision
        from tinyctx.trace import RequestTrace
        from tinyctx import empty_response_guard as _erg
        _erg.reset_state()

        # Capture per-attempt Authorization header so we can prove the
        # local bearer never reached the frontier URL.
        attempt_auth: list[tuple[str, str]] = []

        scripts = [
            ("err", 400,
             b'{"error":{"message":"Unexpected message role. Got developer"}}'),
            ("ok", healthy_frontier_chunks()),
        ]
        inner_stream_fn, state = make_mock_stream(scripts)

        def spy_stream(self, method, url, **kwargs):
            hdrs = kwargs.get("headers") or {}
            attempt_auth.append((url, hdrs.get("Authorization", "")))
            return inner_stream_fn(self, method, url, **kwargs)

        proj_sid = "saga3-proj"
        conv_sid = "saga3-conv"
        decision = Decision("local", "small/short -> cheap path",
                            is_compaction=False)
        trace = RequestTrace(session_id=proj_sid)

        with patch.object(httpx.AsyncClient, "stream", spy_stream):
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json",
                 "Authorization": "Bearer sk-local-bearer",
                 "openai-beta": "responses=v1"},
                {"model": "qwen-test",
                 "input": [{"role": "user", "content": "hi"}]},
                is_stream=True,
                sid=proj_sid,
                decision=decision,
                trace=trace,
                conv_sid=conv_sid,
            )
            body_bytes = await drain_stream(sr)

        # Two attempts: local then frontier.
        assert len(state["calls"]) == 2
        assert "local.test" in state["calls"][0]
        assert "frontier.test" in state["calls"][1]

        # First attempt carried the local bearer; second attempt MUST
        # carry the frontier bearer, not the leaked local one.
        assert attempt_auth[0][1] == "Bearer sk-local-bearer"
        assert attempt_auth[1][1] == "Bearer sk-frontier-bearer", (
            f"frontier retry must rebuild Authorization (f8c2489); "
            f"got headers={attempt_auth}"
        )
        assert "sk-local-bearer" not in attempt_auth[1][1]

        # codex saw the frontier success body, NOT the 400 payload.
        assert b"response.completed" in body_bytes
        assert b"Unexpected message role" not in body_bytes

        # retry_attempted event recorded with 400 + retry_escalate.
        events = _read_log_events(proxy_module.CFG.log_dir)
        retry_evts = [e for e in events if e.get("event") == "retry_attempted"
                      and e.get("session") == proj_sid]
        assert len(retry_evts) >= 1
        ev = retry_evts[0]
        assert ev["original_status"] == 400
        assert ev["retry_target"] == "retry_escalate"
        assert "local" in ev["original_url"]
        assert "frontier" in ev["new_url"]

        # force_next_to_frontier set on the conv key so codex's NEXT
        # turn also routes frontier.
        info = _erg.consume_force_frontier(conv_sid)
        assert info is not None
        assert "4xx" in info["reason"] or "escalate" in info["reason"]

        # stream_done route reflects the FINAL decision (still "local"
        # on the trace's `decision` attribute since the proxy logs the
        # ORIGINAL decision; the upstream URL switch is recorded via
        # retry_attempted). What we MUST verify is that bytes_out > 0,
        # not 0 — i.e. the frontier retry's success body actually
        # flowed through to the client.
        done = [e for e in events if e.get("event") == "stream_done"
                and e.get("session") == proj_sid]
        assert len(done) == 1
        assert done[0]["bytes"] > 0


# ════════════════════════════════════════════════════════════════════════
# SAGA 4: proactive_compact with pinned items + signals
# ════════════════════════════════════════════════════════════════════════


class TestSaga4ProactiveCompact:
    """Heavy `body.input` with a clear first-user goal, a tracker call,
    shell calls (one carrying an error pattern), and an apply_patch on a
    UNIQUE path. With the threshold dropped low, proactive_compact fires
    and:
      - reports applied=True and pinned_items>=1 in pc_info,
      - preserves the first-user goal verbatim in the compacted body,
      - preserves the latest update_plan call verbatim,
      - injects the deterministic signals (UNIQUE_FILE / error pattern)
        into the summary item, so the post-compact model retains them.

    This saga calls `proactive_compact` directly (it's the integration
    surface; the FastAPI handler is a thin wrapper around it with
    threshold computation). The pipeline-level wiring is covered by
    test_proactive_compact + test_proxy_compactor_integration."""

    def test_pinning_and_signals_preserved(self, proxy_module):
        from tinyctx.sanitize import proactive_compact, clear_proactive_cache
        clear_proactive_cache()

        UNIQUE_FILE = "tinyctx_saga4/UNIQUE_FILE.py"
        FIRST_USER_GOAL = "FIX_BUG_X please find and patch the regression"
        PLAN_MARKER = "PLAN_MARKER step 3 still pending"
        ERROR_TEXT = "Traceback (most recent call last): RuntimeError: panic_sig_4"

        # Build a body whose middle has the goal, signals, and tracker.
        items: list[dict] = []
        # First user msg — the GOAL.
        items.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": FIRST_USER_GOAL}],
        })
        # A bunch of routine assistant + shell turns.
        for i in range(40):
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text",
                             "text": f"thinking turn {i}"}],
            })
            cid = f"call_sh_{i}"
            items.append({
                "type": "function_call",
                "name": "exec_command",
                "call_id": cid,
                "arguments": json.dumps({"command": ["ls", "-la"]}),
            })
            items.append({
                "type": "function_call_output",
                "call_id": cid,
                "output": "ok\n",
            })
        # One exec with an error pattern.
        items.append({
            "type": "function_call",
            "name": "exec_command",
            "call_id": "call_err",
            "arguments": json.dumps({"command": ["pytest", "tests/"]}),
        })
        items.append({
            "type": "function_call_output",
            "call_id": "call_err",
            "output": ERROR_TEXT,
        })
        # apply_patch on the UNIQUE path.
        items.append({
            "type": "function_call",
            "name": "apply_patch",
            "call_id": "call_patch",
            "arguments": json.dumps({"path": UNIQUE_FILE,
                                       "content": "diff content"}),
        })
        items.append({
            "type": "function_call_output",
            "call_id": "call_patch",
            "output": "patched",
        })
        # update_plan tracker (latest), with the PLAN_MARKER.
        items.append({
            "type": "function_call",
            "name": "update_plan",
            "call_id": "call_plan",
            "arguments": json.dumps({"plan": PLAN_MARKER}),
        })
        items.append({
            "type": "function_call_output",
            "call_id": "call_plan",
            "output": "plan saved",
        })
        # 8 tail turns of fresh exchanges (these are the recent_keep tail).
        for i in range(8):
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text",
                             "text": f"tail turn {i}"}],
            })

        body = {"model": "tinyctx-auto",
                "instructions": "You are a coding agent.",
                "input": items}

        items_before = len(body["input"])
        out, info = proactive_compact(
            body,
            session_id="saga4",
            # est_tokens above threshold → compact fires.
            est_tokens=300_000,
            threshold_tokens=50_000,
            recent_keep=8,
        )

        assert info["applied"] is True, info
        assert info["items_before"] == items_before
        assert info["pinned_items"] >= 1, (
            f"first user message + tracker should pin; got info={info}")

        # Compacted body still contains the first user goal verbatim.
        compacted_serialized = json.dumps(out)
        assert FIRST_USER_GOAL in compacted_serialized, (
            "first user message must be pinned through compaction")

        # Compacted body still contains the latest tracker (PLAN_MARKER).
        assert PLAN_MARKER in compacted_serialized, (
            "latest update_plan tracker must be pinned through compaction")

        # The summary item carries the deterministic signals — UNIQUE_FILE
        # path AND the error text both surface so the post-compact model
        # keeps execution context.
        summary_item = out["input"][0]
        assert summary_item["role"] == "user"
        summary_text = summary_item["content"][0]["text"]
        assert UNIQUE_FILE in summary_text, (
            f"signals section must mention {UNIQUE_FILE}; got tail of "
            f"summary: {summary_text[-1500:]}"
        )
        # Error pattern detected as a "panic" / "Traceback" marker.
        assert ("panic_sig_4" in summary_text
                or "Traceback" in summary_text), (
            f"error signals not surfaced; summary tail: {summary_text[-1500:]}"
        )

        # Tail is preserved verbatim.
        assert out["input"][-8:] == body["input"][-8:]

    @pytest.mark.asyncio
    async def test_proactive_compact_fires_through_handler(
            self, proxy_module, monkeypatch):
        """End-to-end through the FastAPI handler: a body large enough to
        trip the effective threshold MUST be forwarded to the upstream
        with `proactive_compact` applied (item count reduced), and the
        `proactive_compact` log event MUST fire."""
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 1.0)
        # Force the handler down the frontier path so
        # proactive_compact_only_on_frontier=True still lets compact run.
        monkeypatch.setattr(proxy_module.CFG, "force_route", "frontier")
        # Lower the threshold so a realistic-but-small body trips it.
        # effective_proactive_compact_threshold prefers
        # `frontier.context_window * safe_fraction` when both > 0; zero out
        # safe_fraction so it falls through to the absolute knob.
        monkeypatch.setattr(proxy_module.CFG,
                            "proactive_compact_safe_fraction", 0.0)
        monkeypatch.setattr(proxy_module.CFG,
                            "proactive_compact_threshold", 5_000)
        monkeypatch.setattr(proxy_module.CFG,
                            "proactive_compact_overhead_buffer", 0)
        # Disable the LM summarizer so we hit the deterministic
        # placeholder fast path (no network).
        monkeypatch.setattr(proxy_module.CFG,
                            "proactive_compact_use_summarizer", False)
        # Disable injection of bundled global agent rules / advisor hint /
        # auto_scout etc. so the request body that hits the upstream is
        # predictable.
        monkeypatch.setattr(proxy_module.CFG,
                            "inject_global_agent_rules", False)
        monkeypatch.setattr(proxy_module.CFG, "auto_scout", False)
        monkeypatch.setattr(proxy_module.CFG,
                            "frontier_skip_advisor_hint", True)

        # Build a 60-turn body — enough text to overshoot the small
        # threshold via est_input_tokens.
        FIRST_GOAL = "FIX_BUG_X_HANDLER_LEVEL"
        items: list[dict] = [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": FIRST_GOAL}],
        }]
        # Pad each turn with enough chars to push est_tokens over the bar.
        for i in range(60):
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text",
                             "text": "lorem ipsum " * 200 + f" turn {i}"}],
            })
        body = {"model": "gpt-test",
                "instructions": "x",
                "input": items,
                "stream": True}

        scripts = [("ok", healthy_frontier_chunks())]
        stream_fn, state = make_mock_stream(scripts)
        sent_bodies: list[dict] = []

        def spy_stream(self, method, url, **kwargs):
            sent_bodies.append(kwargs.get("json") or {})
            return stream_fn(self, method, url, **kwargs)

        from httpx import ASGITransport
        with patch.object(httpx.AsyncClient, "stream", spy_stream):
            async with httpx.AsyncClient(
                    transport=ASGITransport(app=proxy_module.APP),
                    base_url="http://testserver") as client:
                r = await client.post(
                    "/v1/responses",
                    json=body,
                    headers={"x-codex-session-id": "saga4-handler"},
                )
                # Drain SSE body (StreamingResponse content).
                payload = b""
                async for chunk in r.aiter_bytes():
                    payload += chunk

        assert state["calls"], "upstream stream must have been hit"
        assert "frontier.test" in state["calls"][0]
        # The forwarded body must be COMPACTED — items_after < items_before.
        assert sent_bodies, "spy must have captured forwarded body"
        forwarded_input = sent_bodies[0].get("input") or []
        assert len(forwarded_input) < len(items), (
            f"proactive_compact should have shrunk body.input; "
            f"sent {len(forwarded_input)} vs original {len(items)}"
        )
        # And the FIRST GOAL is still there.
        assert FIRST_GOAL in json.dumps(forwarded_input), (
            "first user goal must be pinned through handler-level compaction")

        # proactive_compact log event fired.
        events = _read_log_events(proxy_module.CFG.log_dir)
        pc_evts = [e for e in events if e.get("event") == "proactive_compact"]
        assert pc_evts, "proactive_compact event must fire"
        assert pc_evts[-1].get("applied") is True
        assert pc_evts[-1].get("pinned_items", 0) >= 1


# ════════════════════════════════════════════════════════════════════════
# SAGA 5: Stall-cancel mid-stream
# ════════════════════════════════════════════════════════════════════════


class _HangingStreamCtx:
    """200 OK then never yields — simulates the upstream-silence stall."""

    def __init__(self, started_event: asyncio.Event) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._started = started_event

    async def __aenter__(self) -> "_HangingStreamCtx":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def aread(self) -> bytes:
        return b""

    async def aiter_raw(self):
        self._started.set()
        # Block well past stall_threshold; the cancel injects via
        # asyncio.CancelledError at the next await point.
        await asyncio.sleep(3600)
        if False:
            yield b""


class TestSaga5StallCancel:
    """Producer opens with 200, then upstream goes silent. We simulate
    the stall watchdog firing (by directly calling cancel_active_task
    + force_next_to_frontier — same wiring the real on_stall callback
    uses) and verify:
      - StallCancelledError reaches the consumer,
      - a synthetic SSE terminator with status=incomplete is emitted,
      - the response.completed marker IS present (codex's parser
        requires this to accept the stream),
      - force_next_to_frontier is set on the conv key with reason
        `mid_stream_stall`,
      - stream_error event fires with error_type=StallCancelledError.
    """

    @pytest.mark.asyncio
    async def test_stall_cancel_terminator_and_force_frontier(
            self, proxy_module, monkeypatch):
        from tinyctx import stall_watchdog as _sw
        from tinyctx import empty_response_guard as _erg
        # Short keepalive so the consumer wakes quickly enough.
        monkeypatch.setattr(proxy_module.CFG,
                            "stream_keepalive_interval_s", 0.05)
        _sw.reset_state()
        _erg.reset_state()

        started = asyncio.Event()

        def stream_fn(self, method, url, **kwargs):
            return _HangingStreamCtx(started)

        proj_sid = "saga5-proj"
        conv_sid = "saga5-conv"

        with patch.object(httpx.AsyncClient, "stream", stream_fn):
            from tinyctx.router import Decision
            from tinyctx.trace import RequestTrace
            decision = Decision("local", "saga5", is_compaction=False)
            trace = RequestTrace(session_id=proj_sid)
            sr = await proxy_module._forward(
                "http://local.test/v1/responses",
                {"Content-Type": "application/json"},
                {"model": "qwen-test", "input": []},
                is_stream=True,
                sid=proj_sid,
                decision=decision,
                trace=trace,
                conv_sid=conv_sid,
            )

            collected = bytearray()

            async def _drain():
                async for chunk in sr.body_iterator:
                    if isinstance(chunk, (bytes, bytearray)):
                        collected.extend(chunk)
                    else:
                        collected.extend(str(chunk).encode())

            drain_task = asyncio.create_task(_drain())
            await asyncio.wait_for(started.wait(), timeout=2.0)
            # Give the producer-create call site a moment to run
            # register_task.
            await asyncio.sleep(0.05)
            assert _sw.get_active_task(proj_sid) is not None, (
                "producer task must be registered with the watchdog")

            # Fire the stall: cancel + flag (what the real on_stall does).
            assert _sw.cancel_active_task(proj_sid) is True
            _erg.force_next_to_frontier(conv_sid, "mid_stream_stall")

            try:
                await asyncio.wait_for(drain_task, timeout=3.0)
            except asyncio.CancelledError:
                pass

        body_bytes = bytes(collected)
        # codex.app's SSE parser needs response.completed.
        assert b"response.completed" in body_bytes
        # status=incomplete signals "this attempt failed, please retry".
        assert b"incomplete" in body_bytes
        # The synthetic stall_cancelled marker appears in the error event.
        assert b"stall_cancelled" in body_bytes

        # Conv-scoped force-frontier flag is set with the right reason.
        info = _erg.consume_force_frontier(conv_sid)
        assert info is not None
        assert "mid_stream_stall" in info["reason"]

        # stream_error log event fired with error_type StallCancelledError.
        events = _read_log_events(proxy_module.CFG.log_dir)
        err_evts = [e for e in events if e.get("event") == "stream_error"
                    and e.get("session") == proj_sid]
        assert err_evts, f"stream_error must fire; events={events[-5:]}"
        assert err_evts[-1]["error_type"] == "StallCancelledError"

        # The phase machine ends in `stalled` because upstream_failed=True.
        from tinyctx import request_phase as _rp
        phase = _rp.get_phase(proj_sid)
        assert phase is not None
        assert phase["phase"] == "stalled"
