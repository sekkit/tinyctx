"""Tests for the multi-subagent compactor.

We stand up a fake OpenAI-compat /chat/completions backend that responds
differently per role (detected via the system prompt content), so we can
verify:
  - all 3 roles fire in parallel
  - the judge fires after with all 3 drafts in its user prompt
  - the SSE wrapper emits the expected event sequence
  - fallbacks (judge fails, all roles fail) behave as designed
"""
from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from tinyctx.compactor import (
    ROLES,
    _flatten_history,
    _gather_drafts,
    build_responses_api_payload,
    build_responses_api_sse,
    compact_with_debate,
)
from tinyctx.config import BackendCfg


def _spawn_fake(received: list[dict], scripts: dict[str, str]) -> tuple[ThreadingHTTPServer, int]:
    """scripts maps a substring of system prompt -> response content."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n))
            received.append(body)
            sys_msg = ""
            for m in body.get("messages", []):
                if m.get("role") == "system":
                    sys_msg = m.get("content", "")
                    break
            payload_text = "(no match)"
            for hint, response in scripts.items():
                if hint.lower() in sys_msg.lower():
                    payload_text = response
                    break
            out = json.dumps({
                "choices": [{"message": {"content": payload_text}}]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def _backend(port: int, *, model: str = "qwen-fake",
             api_key_env: str | None = None) -> BackendCfg:
    return BackendCfg(
        base_url=f"http://127.0.0.1:{port}/v1",
        model=model,
        wire_api="chat",
        timeout_s=10.0,
        api_key_env=api_key_env,
    )


def test_local_call_sends_authorization_when_api_key_env_set():
    """Regression: compactor._local_call MUST send an Authorization header
    when the backend declares an api_key_env. Without this, every
    DeepSeek-style hosted backend returns 401, all 3 role drafts fail,
    and codex's auto-compact silently degrades to a 43-char placeholder
    that wipes out the model's memory.

    Live trace 16:32:22 confirmed this bug surface as
    'earlier task details were compacted out' in codex.app's UI.
    """
    import json as _json
    import os as _os
    received: list[dict] = []

    class _AuthCheck(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = _json.loads(self.rfile.read(n))
            received.append({
                "authorization": self.headers.get("Authorization", ""),
                "model": body.get("model"),
            })
            out = _json.dumps({"choices":[{"message":{"content":"ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _AuthCheck)
    Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        env_var = "TEST_TINYCTX_FAKE_API_KEY"
        _os.environ[env_var] = "sk-fake-deadbeef"
        backend = _backend(port, api_key_env=env_var)

        async def go():
            import httpx
            from tinyctx.compactor import _local_call
            async with httpx.AsyncClient() as client:
                return await _local_call(client, backend, "system", "user",
                                          max_tokens=50)

        # Use new_event_loop pattern to match other tests in this file
        # (asyncio.run() in Py3.9 closes the loop and breaks subsequent
        # tests that call get_event_loop()).
        result = asyncio.new_event_loop().run_until_complete(go())
        assert result == "ok"
        assert len(received) == 1
        assert received[0]["authorization"] == "Bearer sk-fake-deadbeef", (
            f"compactor must send Bearer auth, got: {received[0]['authorization']!r}"
        )
    finally:
        httpd.shutdown()
        _os.environ.pop("TEST_TINYCTX_FAKE_API_KEY", None)


# ---------------------------------------------------------------- unit


def test_flatten_history_strips_empty_and_truncates():
    body = {
        "input": [
            {"role": "user", "content": "what's up"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "x" * 200_000},
        ]
    }
    blob = _flatten_history(body, max_chars=1000)
    assert "what's up" in blob
    assert "[middle truncated by tinyctx compactor]" in blob
    assert len(blob) <= 2000


def test_responses_api_payload_shape():
    p = build_responses_api_payload("hello world", "m")
    assert p["object"] == "response"
    assert p["status"] == "completed"
    assert p["model"] == "m"
    assert p["output"][0]["content"][0]["text"] == "hello world"


def test_sse_emits_full_event_sequence():
    raw = build_responses_api_sse("merged summary", "m")
    text = raw.decode()
    for evt in ("response.created", "response.output_item.added",
                "response.output_text.delta", "response.output_text.done",
                "response.completed"):
        assert evt in text, f"missing event: {evt}"
    assert "merged summary" in text


# ----------------------------------------------------------- integration


def test_three_roles_run_in_parallel_and_judge_merges():
    received: list[dict] = []
    # Use "you are a/an X" hints — these are unique to each role's prompt
    # because the cross-references inside one prompt only mention bare role
    # names, never the "you are an X" phrasing.
    scripts = {
        "you are an archaeologist": "FACTS: file=src/a.py, decision=use jwt",
        "you are a narrator":       "STORY: user wanted auth, we tried jwt",
        "you are an enumerator":    "ARTIFACTS:\n- src/a.py\n- ran pytest",
        "you are a handoff editor":
            "## What we are doing and why\nauth via jwt\n"
            "## Files & decisions\nsrc/a.py\n"
            "## Commands & outcomes\npytest\n"
            "## Open issues / next steps\nnone",
    }
    httpd, port = _spawn_fake(received, scripts)
    try:
        body = {
            "input": [
                {"role": "user", "content": "set up auth"},
                {"role": "assistant", "content": "trying jwt"},
                {"role": "user", "content": "compact this conversation"},
            ]
        }
        backend = _backend(port)
        summary, telemetry, structured = asyncio.get_event_loop().run_until_complete(
            compact_with_debate(body, backend)
        )
        # 3 role calls + 1 judge call
        assert len(received) == 4, f"expected 4 calls, got {len(received)}: {received}"
        # Judge prompt must mention each role's draft
        judge_call = next(
            r for r in received
            if "you are a handoff editor" in r["messages"][0]["content"].lower()
        )
        judge_user = judge_call["messages"][1]["content"]
        assert "<archaeologist>" in judge_user
        assert "<narrator>" in judge_user
        assert "<enumerator>" in judge_user
        # Final summary is the judge's output
        assert "What we are doing" in summary
        assert telemetry["outcome"] == "judged"
        assert set(telemetry["drafts_completed"]) == {"archaeologist",
                                                      "narrator", "enumerator"}
    finally:
        httpd.shutdown()


def test_role_failure_lets_judge_proceed_with_survivors():
    """If one role's HTTP call fails, the judge still merges the surviving
    two."""
    received: list[dict] = []
    scripts = {
        "you are an archaeologist": "ARCHAEO_OK",
        "you are a narrator":       "NARRATOR_OK",
        "you are an enumerator":    "ENUM_OK",
        "you are a handoff editor": "JUDGE_MERGED",
    }
    httpd, port = _spawn_fake(received, scripts)

    # Patch _local_call to raise for archaeologist.
    from tinyctx import compactor as cm
    orig = cm._local_call

    async def patched(client, backend, system_prompt, user_prompt, **kw):
        # Match only the archaeologist's own role prompt, not the judge's
        # cross-reference to ARCHAEOLOGIST.
        if "you are an archaeologist" in system_prompt.lower():
            raise RuntimeError("simulated failure")
        return await orig(client, backend, system_prompt, user_prompt, **kw)

    cm._local_call = patched
    try:
        body = {"input": [{"role": "user", "content": "test"}]}
        summary, tele, structured = asyncio.new_event_loop().run_until_complete(
            cm.compact_with_debate(body, _backend(port))
        )
        # judge ran with 2 survivors
        assert tele["outcome"] == "judged"
        assert "archaeologist" not in tele["drafts_completed"]
        assert {"narrator", "enumerator"}.issubset(set(tele["drafts_completed"]))
        assert summary == "JUDGE_MERGED"
    finally:
        cm._local_call = orig
        httpd.shutdown()


def test_judge_failure_falls_back_to_concat():
    received: list[dict] = []
    scripts = {
        "you are an archaeologist": "A",
        "you are a narrator":       "B",
        "you are an enumerator":    "C",
        "you are a handoff editor": "JUDGE",  # never reached because judge fails
    }
    httpd, port = _spawn_fake(received, scripts)

    from tinyctx import compactor as cm
    orig = cm._local_call

    async def patched(client, backend, system_prompt, user_prompt, **kw):
        if "you are a handoff editor" in system_prompt.lower():
            raise RuntimeError("judge fail")
        return await orig(client, backend, system_prompt, user_prompt, **kw)

    cm._local_call = patched
    try:
        body = {"input": [{"role": "user", "content": "x"}]}
        summary, tele, structured = asyncio.new_event_loop().run_until_complete(
            cm.compact_with_debate(body, _backend(port))
        )
        assert tele["outcome"] == "judge_failed_concat"
        for txt in ("archaeologist", "narrator", "enumerator", "A", "B", "C"):
            assert txt in summary, f"fallback merge missing {txt!r}"
    finally:
        cm._local_call = orig
        httpd.shutdown()


def test_all_failures_returns_marker():
    # No fake server — all calls will fail.
    from tinyctx import compactor as cm
    body = {"input": [{"role": "user", "content": "x"}]}
    backend = BackendCfg(base_url="http://127.0.0.1:1", model="x",
                         wire_api="chat", timeout_s=0.5)
    summary, tele, _structured = asyncio.new_event_loop().run_until_complete(
        cm.compact_with_debate(body, backend)
    )
    assert tele["outcome"] == "all_failed"
    assert "tinyctx compactor" in summary


# ----------------------------------------------------- #4 structured output


def test_parse_judge_output_extracts_fenced_json():
    from tinyctx.compactor import parse_judge_output
    text = (
        "## What we are doing and why\nset up auth\n\n"
        "## Files & decisions\nsrc/auth.py\n\n"
        '```json\n{"compartments": [{"name": "auth-setup", "topic": "JWT", '
        '"summary": "X", "files": ["src/auth.py"]}], '
        '"facts": [{"claim": "secret in .env", "evidence": "user said"}], '
        '"open_questions": ["test coverage?"]}\n```\n'
    )
    md, structured = parse_judge_output(text)
    assert "What we are doing" in md
    assert "```json" not in md  # fence stripped
    assert len(structured["compartments"]) == 1
    assert structured["compartments"][0]["name"] == "auth-setup"
    assert structured["facts"][0]["claim"] == "secret in .env"
    assert structured["open_questions"] == ["test coverage?"]


def test_parse_judge_output_falls_back_to_bare_object():
    from tinyctx.compactor import parse_judge_output
    text = (
        "## section\nbody\n\n"
        '{"compartments": [], "facts": [{"claim": "x", "evidence": "y"}], '
        '"open_questions": []}'
    )
    md, structured = parse_judge_output(text)
    assert structured["facts"][0]["claim"] == "x"
    assert "section" in md


def test_parse_judge_output_degrades_gracefully_when_no_json():
    from tinyctx.compactor import parse_judge_output
    text = "just a markdown summary, no JSON block"
    md, structured = parse_judge_output(text)
    assert md == text
    assert structured["compartments"] == []
    assert structured["facts"] == []


# ---------------------------------------- #2 pristine recomputation guard


def test_pristine_recomputation_guard():
    """Compactor must never feed its own output back through the role
    drafts. Codex naturally provides pristine history every turn, but if a
    future incremental compactor regresses, this test flags it.

    We simulate by running the compactor twice on the SAME body and
    asserting the second call's user prompts are not contaminated by the
    first call's role outputs.
    """
    received: list[dict] = []
    scripts = {
        "you are an archaeologist": "FACTS-from-archaeologist-v1",
        "you are a narrator":       "STORY-from-narrator-v1",
        "you are an enumerator":    "ARTIFACTS-from-enumerator-v1",
        "you are a handoff editor": "## merged\n```json\n{}\n```",
    }
    httpd, port = _spawn_fake(received, scripts)
    try:
        body = {"input": [
            {"role": "user", "content": "build something"},
            {"role": "assistant", "content": "doing it"},
        ]}
        backend = _backend(port)
        loop = asyncio.new_event_loop()
        s1, _, _ = loop.run_until_complete(compact_with_debate(body, backend))
        # Reset call log; run again with the IDENTICAL body.
        received.clear()
        s2, _, _ = loop.run_until_complete(compact_with_debate(body, backend))
        # The pristine invariant applies to ROLE drafts (archaeologist,
        # narrator, enumerator) — they must see only the original
        # conversation history, never role-output text. The judge
        # legitimately receives the three role drafts as its user prompt;
        # exclude it from the contamination check.
        role_calls = [
            c for c in received
            if "you are a handoff editor" not in c["messages"][0]["content"].lower()
        ]
        assert len(role_calls) == 3, f"expected 3 role calls, got {len(role_calls)}"
        for call in role_calls:
            user_msg = call["messages"][1]["content"]
            assert "FACTS-from-archaeologist-v1" not in user_msg, \
                "role-draft input contaminated by prior compactor output"
            assert "STORY-from-narrator-v1" not in user_msg
            assert "ARTIFACTS-from-enumerator-v1" not in user_msg
        # And the two summaries are identical (deterministic on identical input).
        assert s1 == s2
    finally:
        httpd.shutdown()


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
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
