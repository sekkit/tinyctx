"""Deterministic Responses fixture replay for tool-call guardrails.

These fixtures are the first tinyctx-native equivalent of forge's eval
culture: replay a known upstream shape and assert the canonical Responses
wire semantics tinyctx must preserve.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyctx.tool_call_translator import StreamTranslator, rebuild_response


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "responses"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _message_texts(response: dict) -> list[str]:
    texts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                texts.append(str(content.get("text") or ""))
    return texts


def _function_calls(response: dict) -> list[dict]:
    return [
        item for item in (response.get("output") or [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def _sse_delta(delta: str, seq: int) -> bytes:
    payload = {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
        "sequence_number": seq,
    }
    return (
        "event: response.output_text.delta\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


def _parse_sse_events(rendered: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in rendered.split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if event and data:
            events.append((event, json.loads(data)))
    return events


def _is_ordered_subsequence(needles: list[str], haystack: list[str]) -> bool:
    pos = 0
    for item in haystack:
        if pos < len(needles) and item == needles[pos]:
            pos += 1
    return pos == len(needles)


@pytest.mark.parametrize(
    "fixture_path",
    sorted((FIXTURE_ROOT / "sync").glob("*.json")),
    ids=lambda p: p.stem,
)
def test_sync_response_fixture_replay(fixture_path: Path):
    fixture = _load(fixture_path)

    valid = fixture.get("valid_tool_names")
    out = rebuild_response(
        fixture["upstream"],
        valid_tool_names=set(valid) if valid is not None else None,
    )
    calls = _function_calls(out)
    texts = "\n".join(_message_texts(out))
    expected = fixture["expected"]

    assert [c.get("name") for c in calls] == expected["function_call_names"]
    for call, args in zip(calls, expected.get("function_call_args", [])):
        assert json.loads(call.get("arguments") or "{}") == args
    for needle in expected.get("message_text_contains", []):
        assert needle in texts


@pytest.mark.parametrize(
    "fixture_path",
    sorted((FIXTURE_ROOT / "stream").glob("*.json")),
    ids=lambda p: p.stem,
)
def test_stream_response_fixture_replay(fixture_path: Path):
    fixture = _load(fixture_path)
    translator = StreamTranslator()
    out: list[bytes] = []

    for seq, delta in enumerate(fixture["deltas"], start=1):
        out.extend(translator.feed(_sse_delta(delta, seq)))

    rendered = b"".join(out).decode("utf-8")
    events = _parse_sse_events(rendered)
    expected = fixture["expected"]
    for needle in expected.get("contains", []):
        assert needle in rendered
    for needle in expected.get("not_contains", []):
        assert needle not in rendered
    if "event_subsequence" in expected:
        assert _is_ordered_subsequence(
            expected["event_subsequence"],
            [event for event, _ in events],
        )
