from __future__ import annotations

import asyncio
import unittest.mock as mock

from tinyctx.advisor_continuation import (
    PendingWork,
    consume_pending_work,
    extract_pending_work_from_outgoing_sse,
    inject_pending_work_into_body,
    reset_state,
    store_pending_work,
)
from tinyctx.guards import AdvisorContinuationGuard, GuardContext


def test_store_and_consume_pending_work():
    reset_state("sid1")
    assert store_pending_work("sid1", "1. Inspect files\n2. Port renderer", source="advisor") is True
    got = consume_pending_work("sid1")
    assert got is not None
    assert "Inspect files" in got.work_text
    assert got.source == "advisor"
    assert consume_pending_work("sid1") is None


def test_inject_pending_work_into_body():
    body = {
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hello"}]}
        ]
    }
    pending = PendingWork("1. Continue\n2. Verify", "advisor", 1.0)
    new_body, ok = inject_pending_work_into_body(body, pending)
    assert ok is True
    assert new_body is not body
    assert len(new_body["input"]) == 2
    last = new_body["input"][-1]
    assert last["role"] == "user"
    assert "Continue with this work" in last["content"][0]["text"]
    assert "1. Continue" in last["content"][0]["text"]


def test_guard_injects_and_consumes_pending_work():
    reset_state("sid2")
    assert store_pending_work("sid2", "1. Read official-like-compose-xr", source="stream") is True
    ctx = GuardContext(
        body={
            "model": "test",
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "resume"}]}],
        },
        proj_sid="sid2",
        conv_sid="sid2",
        turn_count=3,
    )
    g = AdvisorContinuationGuard()
    result = g.apply(ctx)
    assert result.fired is True
    assert result.body_mutated is True
    assert "advisor continuation" in result.reason
    assert len(ctx.body["input"]) == 2
    assert "official-like-compose-xr" in ctx.body["input"][-1]["content"][0]["text"]
    again = g.apply(ctx)
    assert again.fired is False


def test_extract_pending_work_from_outgoing_sse_requires_advisor_markers():
    text = 'event: response.output_item.done\ndata: {"type":"response.output_item.done","item":{"output":"1. Do A\\n2. Do B"}}\n\n'
    assert asyncio.run(extract_pending_work_from_outgoing_sse(text)) == ""


def test_extract_pending_work_from_outgoing_sse_parses_observed_shape():
    text = (
        'event: response.output_item.done\n'
        'data: {"type":"response.output_item.done","item":{"name":"ask_advisor","namespace":"mcp__advisor__",'
        '"output":"1. Read official-like-compose-xr\\n2. Port native commit path\\nRisks: keep GLB uncompressed"}}\n\n'
    )
    mock_resp = mock.MagicMock()
    mock_resp.raise_for_status = mock.MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "YES"}}]}
    mock_client = mock.AsyncMock()
    mock_client.post = mock.AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mock.AsyncMock(return_value=None)
    with mock.patch("httpx.AsyncClient", return_value=mock_client):
        got = asyncio.run(extract_pending_work_from_outgoing_sse(
            text,
            local_base_url="http://localhost:1234",
            local_model="test-model",
        ))
    assert "official-like-compose-xr" in got
    assert "Port native commit path" in got
