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


def test_auto_answer_enabled_by_default():
    """TINYCTX_AUTO_USER_INPUT defaults to ON. With env unset and no
    advisor mock, the function attempts a real advisor call and returns
    None on its httpx failure (no advisor reachable in unit tests).
    Either way, the function ATTEMPTS to run rather than short-circuiting."""
    import os as _os
    import tinyctx.advisor as _adv
    from tinyctx.tool_call_translator import _try_auto_answer_user_input
    saved_env = _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
    saved_adv = _adv.call_advisor
    captured = {}
    def fake(question, context="", previous_attempts=""):
        captured["called"] = True
        return {"text": "Pick: A — default-on test.",
                "usage": None, "error": None}
    _adv.call_advisor = fake
    try:
        out = _try_auto_answer_user_input(json.dumps({
            "questions": [{"header": "A or B?", "options": ["A", "B"]}]
        }))
        # default is on, so advisor SHOULD have been consulted
        assert captured.get("called") is True, \
            "default-on means advisor consulted without explicit env=1"
        assert out is not None
        assert "auto-decision" in out
    finally:
        if saved_env is not None:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved_env
        _adv.call_advisor = saved_adv


def test_auto_answer_disabled_with_env_zero():
    """`TINYCTX_AUTO_USER_INPUT=0` must explicitly opt out — function
    returns None without consulting the advisor at all."""
    import os as _os
    import tinyctx.advisor as _adv
    from tinyctx.tool_call_translator import _try_auto_answer_user_input
    saved_env = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    saved_adv = _adv.call_advisor
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "0"
    captured = {}
    def fake(**kw):
        captured["called"] = True
        return {"text": "should not be called",
                "usage": None, "error": None}
    _adv.call_advisor = fake
    try:
        out = _try_auto_answer_user_input(json.dumps({
            "questions": [{"header": "A or B?", "options": ["A", "B"]}]
        }))
        assert out is None
        assert "called" not in captured, \
            "advisor must NOT be consulted when env=0"
    finally:
        if saved_env is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved_env
        _adv.call_advisor = saved_adv


def test_auto_answer_calls_advisor_when_enabled():
    """With TINYCTX_AUTO_USER_INPUT=1, intercept calls advisor and
    returns its text wrapped with an audit-trail marker."""
    import os as _os
    import tinyctx.advisor as _adv
    import tinyctx.tool_call_translator as _tct
    captured = {}

    def fake_call_advisor(question, context="", previous_attempts=""):
        captured["question"] = question
        captured["context"] = context
        return {"text": "Q1: A — fewer vendor deps.", "usage": None,
                "error": None}

    saved_env = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    saved_adv = _adv.call_advisor
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "1"
    _adv.call_advisor = fake_call_advisor
    try:
        out = _tct._try_auto_answer_user_input(json.dumps({
            "questions": [{
                "header": "Pick architecture path",
                "options": ["A: IMU 6DoF", "B: RemoteLoader fix"],
            }]
        }))
        assert out is not None
        assert "advisor auto-decision" in out
        assert "Q1: A" in out
        assert "IMU 6DoF" in captured["question"]
        assert "RemoteLoader" in captured["question"]
    finally:
        if saved_env is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved_env
        _adv.call_advisor = saved_adv


def test_auto_answer_returns_none_on_advisor_failure():
    import os as _os
    import tinyctx.advisor as _adv
    import tinyctx.tool_call_translator as _tct
    saved_env = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    saved_adv = _adv.call_advisor
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "1"
    _adv.call_advisor = lambda **kw: {"text": "", "usage": None,
                                      "error": "network"}
    try:
        out = _tct._try_auto_answer_user_input(json.dumps({
            "questions": [{"header": "x", "options": ["a"]}]
        }))
        assert out is None
    finally:
        if saved_env is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved_env
        _adv.call_advisor = saved_adv


def test_auto_answer_skips_malformed_or_empty():
    import os as _os
    from tinyctx.tool_call_translator import _try_auto_answer_user_input
    saved = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "1"
    try:
        assert _try_auto_answer_user_input("{not json") is None
        assert _try_auto_answer_user_input("") is None
        assert _try_auto_answer_user_input("{}") is None
        assert _try_auto_answer_user_input(
            json.dumps({"questions": []})) is None
    finally:
        if saved is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved


def test_detect_text_choice_chinese_arrow_format():
    """The exact format observed in the live RayNeo session — Chinese
    cue + A/B options with arrow markers — should be detected."""
    from tinyctx.tool_call_translator import _detect_text_choice_prompt
    text = """逆向分析完成。核心发现：
... [analysis] ...

== 重写方案 ==
有两个方向：

**方案 A: IMU 自实现 6DoF（推荐）**
不依赖任何 vendor Binder/AIDL。

**方案 B: RemoteLoader 初始化修复**
在 JNI 中加载 libRayNeoXRRemoteLoader.so

请选择：
A → IMU 自实现 6DoF
B → RemoteLoader 初始化修复
"""
    out = _detect_text_choice_prompt(text)
    assert out is not None, "must detect 请选择 + A → / B → format"
    assert "请选择" in out["header"]
    assert any("IMU" in o for o in out["options"])
    assert any("RemoteLoader" in o for o in out["options"])


def test_detect_text_choice_english_format():
    from tinyctx.tool_call_translator import _detect_text_choice_prompt
    text = """Here's the situation. Two valid paths:

Which option do you prefer?
A. Use the existing transaction system
B. Add a new event-sourcing layer
C. Hybrid (transactions + outbox)
"""
    out = _detect_text_choice_prompt(text)
    assert out is not None
    assert len(out["options"]) >= 3


def test_detect_text_choice_does_not_fire_on_incidental_mentions():
    """Must NOT fire when 'option A' or 'option B' is just mentioned
    in passing inside reasoning prose."""
    from tinyctx.tool_call_translator import _detect_text_choice_prompt
    text = ("I considered option A briefly but option B has better "
            "ergonomics. Option A would require migrating the schema "
            "which is risky. So I went with B. Done — applied the "
            "patches and tests pass.")
    out = _detect_text_choice_prompt(text)
    assert out is None, \
        "incidental mentions of options inside prose must not trigger"


def test_detect_text_choice_requires_two_plus_options():
    """A single 'A. blah' line is just numbered prose, not a choice."""
    from tinyctx.tool_call_translator import _detect_text_choice_prompt
    text = "Please confirm:\nA. proceed with migration"
    assert _detect_text_choice_prompt(text) is None


def test_text_choice_intercept_enabled_by_default():
    """Text-choice intercept also defaults to ON. With env unset and an
    advisor mock, the intercept fires automatically."""
    import os as _os
    import tinyctx.advisor as _adv
    from tinyctx.tool_call_translator import _try_auto_answer_text_choice
    saved_env = _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
    saved_adv = _adv.call_advisor
    _adv.call_advisor = lambda **kw: {
        "text": "Pick: A — default-on text intercept.",
        "usage": None, "error": None,
    }
    try:
        text = "请选择：\nA → 方案一\nB → 方案二"
        out = _try_auto_answer_text_choice(text)
        assert out is not None
        assert "auto-decision" in out
    finally:
        if saved_env is not None:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved_env
        _adv.call_advisor = saved_adv


def test_text_choice_intercept_disabled_with_env_zero():
    import os as _os
    from tinyctx.tool_call_translator import _try_auto_answer_text_choice
    saved = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "0"
    try:
        text = "请选择：\nA → 方案一\nB → 方案二"
        assert _try_auto_answer_text_choice(text) is None
    finally:
        if saved is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved


def test_text_choice_intercept_calls_advisor_when_enabled():
    import os as _os
    import tinyctx.advisor as _adv
    import tinyctx.tool_call_translator as _tct
    captured = {}
    def fake_call_advisor(question, context="", previous_attempts=""):
        captured["q"] = question
        return {"text": "Pick: A — fewer external deps.",
                "usage": None, "error": None}

    saved_env = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    saved_adv = _adv.call_advisor
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "1"
    _adv.call_advisor = fake_call_advisor
    try:
        text = ("做完逆向分析。\n\n请选择：\n"
                "A → IMU 自实现 6DoF\n"
                "B → RemoteLoader 初始化修复")
        out = _tct._try_auto_answer_text_choice(text)
        assert out is not None
        assert "advisor auto-decision" in out
        assert "text-choice intercept" in out
        assert "Pick: A" in out
        assert "IMU" in captured["q"]
        assert "RemoteLoader" in captured["q"]
    finally:
        if saved_env is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved_env
        _adv.call_advisor = saved_adv


def test_text_choice_intercept_returns_none_no_match():
    import os as _os
    import tinyctx.tool_call_translator as _tct
    saved = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "1"
    try:
        # plain finish text with no choice prompt → no advisor call needed
        out = _tct._try_auto_answer_text_choice(
            "Done. All tests pass. Build successful.")
        assert out is None
    finally:
        if saved is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved


def test_translator_finish_intercepts_request_user_input_when_enabled():
    """End-to-end: with env on + advisor mocked, the tool_call entry is
    dropped (so codex's request_user_input UI prompt never fires) and the
    advisor's decision text gets folded into the assistant message text
    instead — codex sees the model just answered directly."""
    import os as _os
    import tinyctx.advisor as _adv
    from tinyctx.tool_call_translator import ChatToResponsesTranslator

    saved_env = _os.environ.get("TINYCTX_AUTO_USER_INPUT")
    saved_adv = _adv.call_advisor
    _os.environ["TINYCTX_AUTO_USER_INPUT"] = "1"
    _adv.call_advisor = lambda **kw: {
        "text": "Q1: B — RemoteLoader fix is closer to vendor intent.",
        "usage": None, "error": None,
    }
    try:
        t = ChatToResponsesTranslator()
        t._model = "deepseek-v4-flash"
        # text first, then a request_user_input function_call, then done
        chunks = [
            ("data: " + json.dumps({
                "id": "x", "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {"content": "Considering options..."},
                    "index": 0,
                }],
            }) + "\n\n").encode(),
            ("data: " + json.dumps({
                "id": "x", "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {"tool_calls": [{
                        "index": 0, "id": "fc_abc", "type": "function",
                        "function": {
                            "name": "request_user_input",
                            "arguments": json.dumps({
                                "questions": [{
                                    "header": "Pick A or B",
                                    "options": ["A: IMU", "B: RemoteLoader"],
                                }]
                            }),
                        },
                    }]},
                    "index": 0,
                }],
            }) + "\n\n").encode(),
            ("data: " + json.dumps({
                "id": "x", "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {}, "finish_reason": "tool_calls", "index": 0,
                }],
            }) + "\n\ndata: [DONE]\n\n").encode(),
        ]
        emitted = b""
        for chunk in chunks:
            for ev in t.feed(chunk):
                emitted += ev
        for ev in t.flush():
            emitted += ev
        text = emitted.decode("utf-8")
        assert "advisor auto-decision" in text
        assert "Q1: B" in text
        # the function_call name was NOT emitted to codex
        assert "request_user_input" not in text
    finally:
        if saved_env is None:
            _os.environ.pop("TINYCTX_AUTO_USER_INPUT", None)
        else:
            _os.environ["TINYCTX_AUTO_USER_INPUT"] = saved_env
        _adv.call_advisor = saved_adv


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
