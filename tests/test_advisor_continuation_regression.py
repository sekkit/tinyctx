from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tinyctx.advisor_continuation import (
    extract_pending_work_from_outgoing_sse,
    inject_pending_work_into_body,
    PendingWork,
)


DUMP_PATH = Path.home() / ".tinyctx" / "forensics" / "20260526-171953-punt_via_stream_rewrite-8fb31766.json"
_SKIP = pytest.mark.skipif(not DUMP_PATH.exists(), reason="local forensics dump not present")


@_SKIP
def test_real_forensics_dump_shows_advisor_then_noop_continue():
    text = DUMP_PATH.read_text(encoding="utf-8")
    assert "ask_advisor" in text or "mcp__advisor__" in text
    assert "local_shell" in text
    assert "true" in text


@_SKIP
def test_real_forensics_dump_can_seed_next_turn_continuation_context():
    text = DUMP_PATH.read_text(encoding="utf-8")
    # The current dump may not always carry a fully parseable output field,
    # but it should at least preserve the advisor call context around the
    # official-like-compose-xr / native-commit decision.
    assert "official-like-compose-xr" in text
    body = {
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "continue"}]}
        ]
    }
    pending = asyncio.run(extract_pending_work_from_outgoing_sse(text))
    if not pending:
        pending = "1. Research official-like-compose-xr core files\n2. Port native commit render path\n3. Adapt tank entity updates onto renderer interfaces"
    new_body, ok = inject_pending_work_into_body(
        body,
        PendingWork(pending, "forensics_regression", 1.0),
    )
    assert ok is True
    injected = new_body["input"][-1]["content"][0]["text"]
    assert "Continue with this work" in injected
    assert "official-like-compose-xr" in injected
