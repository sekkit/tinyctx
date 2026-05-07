"""Tests for tinyctx.tool_call_translator: parse + rebuild_response (non-
streaming) + StreamTranslator (streaming state machine)."""
from __future__ import annotations

import json

from tinyctx.tool_call_translator import (
    StreamTranslator,
    parse_tool_call_block,
    rebuild_response,
)


# ──────────────────────────────────────────────── parse_tool_call_block


def test_parse_single_tool_call_with_one_param():
    text = (
        "<tool_call>\n<function=ls>\n"
        "<parameter=path>\n/tmp\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = parse_tool_call_block(text)
    assert len(out) == 1
    assert out[0]["name"] == "ls"
    args = json.loads(out[0]["arguments"])
    assert args == {"path": "/tmp"}


def test_parse_multiple_params_keeps_order():
    text = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\nfoo.py\n</parameter>\n"
        "<parameter=content>\nprint('hi')\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = parse_tool_call_block(text)
    assert len(out) == 1
    args = json.loads(out[0]["arguments"])
    assert args == {"path": "foo.py", "content": "print('hi')"}


def test_parse_multiple_tool_calls_in_one_text():
    text = (
        "<tool_call>\n<function=ls>\n"
        "<parameter=path>\n/tmp\n</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=ls>\n"
        "<parameter=path>\n/etc\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = parse_tool_call_block(text)
    assert [c["name"] for c in out] == ["ls", "ls"]
    paths = [json.loads(c["arguments"])["path"] for c in out]
    assert paths == ["/tmp", "/etc"]


def test_parse_handles_numeric_and_json_values():
    text = (
        "<tool_call>\n<function=set>\n"
        "<parameter=count>\n42\n</parameter>\n"
        "<parameter=enabled>\ntrue\n</parameter>\n"
        "<parameter=meta>\n{\"k\": 1}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    args = json.loads(parse_tool_call_block(text)[0]["arguments"])
    assert args["count"] == 42
    assert args["enabled"] is True
    assert args["meta"] == {"k": 1}


def test_parse_returns_empty_on_malformed_input():
    assert parse_tool_call_block("plain text") == []
    assert parse_tool_call_block("<tool_call><function=x>") == []   # unclosed
    assert parse_tool_call_block("") == []


# ──────────────────────────────────────────────── rebuild_response


def test_rebuild_response_extracts_into_function_call_item():
    response = {
        "id": "resp_x",
        "object": "response",
        "status": "completed",
        "model": "qwen-fake",
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant",
             "status": "completed",
             "content": [
                 {"type": "output_text",
                  "text": "<tool_call>\n<function=ls>\n"
                          "<parameter=path>\n/tmp\n</parameter>\n"
                          "</function>\n</tool_call>"}
             ]}
        ]
    }
    out = rebuild_response(response)
    items = out["output"]
    # The bare-text-XML message was dropped; one function_call appended.
    assert any(it.get("type") == "function_call" for it in items)
    fc = next(it for it in items if it.get("type") == "function_call")
    assert fc["name"] == "ls"
    assert json.loads(fc["arguments"]) == {"path": "/tmp"}
    # Message item with no remaining text was dropped (no leftover content).
    assert all(it.get("type") != "message" for it in items)


def test_rebuild_response_keeps_pre_call_text():
    """Text BEFORE the <tool_call> block survives as a shortened message
    item alongside the new function_call."""
    response = {
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant",
             "status": "completed",
             "content": [
                 {"type": "output_text",
                  "text": "Sure, let me check that for you. "
                          "<tool_call>\n<function=ls>\n"
                          "<parameter=path>\n/tmp\n</parameter>\n"
                          "</function>\n</tool_call>"}
             ]}
        ]
    }
    out = rebuild_response(response)
    msgs = [it for it in out["output"] if it.get("type") == "message"]
    fcs  = [it for it in out["output"] if it.get("type") == "function_call"]
    assert len(msgs) == 1
    assert "Sure, let me check" in msgs[0]["content"][0]["text"]
    assert "<tool_call>" not in msgs[0]["content"][0]["text"]
    assert len(fcs) == 1


def test_rebuild_response_passes_through_when_no_xml():
    response = {
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "just text"}]}
        ]
    }
    out = rebuild_response(response)
    assert out is response  # same object — no copy when nothing to rewrite


def test_rebuild_response_handles_already_structured():
    """When upstream already returned a function_call item, leave it."""
    response = {
        "output": [
            {"id": "fc_1", "type": "function_call", "name": "ls",
             "arguments": "{\"path\":\"/tmp\"}", "call_id": "fc_1"}
        ]
    }
    out = rebuild_response(response)
    assert out is response
    assert out["output"][0]["type"] == "function_call"


# ──────────────────────────────────────────────── StreamTranslator


def _events_to_str(byte_chunks):
    return b"".join(byte_chunks).decode("utf-8")


def test_stream_translator_passes_through_plain_text():
    t = StreamTranslator()
    chunk = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","item_id":"m1",'
        b'"output_index":0,"content_index":0,"delta":"hello world",'
        b'"sequence_number":1}\n\n'
    )
    out = _events_to_str(t.feed(chunk))
    assert "hello world" in out
    assert "function_call" not in out


def test_stream_translator_emits_function_call_when_block_completes():
    """Feed a fragmented tool-call across multiple deltas; expect proper
    function_call events to come out."""
    t = StreamTranslator()
    deltas = [
        "Sure! ",
        "<tool_call>\n<function=ls>\n",
        "<parameter=path>\n/tmp\n</parameter>\n",
        "</function>\n</tool_call>",
    ]
    out_bytes = []
    for d in deltas:
        chunk = (
            "event: response.output_text.delta\n"
            "data: " + json.dumps({
                "type": "response.output_text.delta",
                "item_id": "m1", "output_index": 0,
                "content_index": 0, "delta": d, "sequence_number": 1,
            }) + "\n\n"
        ).encode("utf-8")
        out_bytes.extend(t.feed(chunk))
    out = _events_to_str(out_bytes)
    # Pre-call plain text was forwarded
    assert "Sure! " in out
    # And we emitted structured function_call events
    assert "response.output_item.added" in out
    assert "function_call" in out
    assert "response.function_call_arguments.delta" in out
    assert "response.function_call_arguments.done" in out
    assert "response.output_item.done" in out
    # The args show up in the synthesized event
    assert "/tmp" in out


def test_stream_translator_passes_through_unknown_events():
    t = StreamTranslator()
    chunk = b"event: response.created\ndata: {\"x\":1}\n\n"
    out = _events_to_str(t.feed(chunk))
    assert "response.created" in out
    assert "{\"x\":1}" in out


def test_stream_translator_drops_xml_from_residual_text():
    """When a tool_call block is sandwiched inside a single delta, the
    head text comes through but the XML block does NOT."""
    t = StreamTranslator()
    delta = ("hi <tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n"
             "</function>\n</tool_call> bye")
    chunk = (
        "event: response.output_text.delta\n"
        "data: " + json.dumps({
            "type": "response.output_text.delta",
            "item_id": "m1", "output_index": 0,
            "content_index": 0, "delta": delta, "sequence_number": 1,
        }) + "\n\n"
    ).encode("utf-8")
    out = _events_to_str(t.feed(chunk))
    # head ('hi') made it through as plain delta
    assert ">hi " in out or '"delta": "hi ' in out or '"delta":"hi ' in out
    # XML block is gone from any output_text.delta
    text_deltas = [line for line in out.split("\n\n")
                   if "response.output_text.delta" in line]
    for td in text_deltas:
        assert "<tool_call>" not in td
    # function_call event was synthesized
    assert "function_call" in out


def test_stream_translator_flush_emits_buffered_tail():
    """If upstream ends mid-buffer with no complete tool_call but with
    plain text, flush re-emits it so the client sees it."""
    t = StreamTranslator()
    chunk = (
        "event: response.output_text.delta\n"
        "data: " + json.dumps({
            "type": "response.output_text.delta",
            "item_id": "m1", "output_index": 0,
            "content_index": 0, "delta": "tail content <tool_call",
            "sequence_number": 1,
        }) + "\n\n"
    ).encode("utf-8")
    # Should NOT emit on feed (we're holding back the partial open-tag).
    feed_out = _events_to_str(t.feed(chunk))
    assert "tail content" not in feed_out
    # Flush emits the held buffer.
    flush_out = _events_to_str(t.flush())
    assert "tail content" in flush_out


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
