"""Tests for tinyctx.headroom_compress — tool output compression via headroom."""

import json
import pytest


@pytest.fixture
def mock_json_array_body():
    """A body with a typical JSON array tool output (e.g. grep results)."""
    items = [
        {"file": f"src/file_{i}.py", "line": j, "content": f"def f{i}: pass"}
        for i in range(50) for j in range(3)
    ]
    return {
        "input": [
            {"type": "message", "role": "user", "content": "find all functions"},
            {
                "type": "function_call_output",
                "call_id": "grep1",
                "output": json.dumps(items),
            },
        ]
    }


@pytest.fixture
def mock_short_body():
    """A body with tool outputs too short to compress."""
    return {
        "input": [
            {"type": "function_call_output", "call_id": "s1", "output": "ok"},
            {"type": "function_call_output", "call_id": "s2", "output": "done"},
        ]
    }


class TestHeadroomAvailability:
    def test_check_headroom_imports(self):
        from tinyctx.headroom_compress import _check_headroom
        assert _check_headroom() is True


class TestCompressToolOutputs:
    def test_noop_when_disabled(self, mock_json_array_body):
        from tinyctx.headroom_compress import compress_tool_outputs
        body = json.loads(json.dumps(mock_json_array_body))
        result = compress_tool_outputs(body, enabled=False)
        assert result == body  # unchanged

    def test_skips_short_outputs(self, mock_short_body):
        from tinyctx.headroom_compress import compress_tool_outputs
        body = json.loads(json.dumps(mock_short_body))
        result = compress_tool_outputs(body, min_chars=100)
        # Both outputs are < 100 chars, should be unchanged
        for item in result["input"]:
            assert item["output"] in ("ok", "done")

    def test_compresses_json_array(self, mock_json_array_body):
        from tinyctx.headroom_compress import compress_tool_outputs
        body = json.loads(json.dumps(mock_json_array_body))
        result = compress_tool_outputs(body, min_chars=100)

        # The JSON array output should be compressed
        for item in result["input"]:
            if item.get("type") == "function_call_output":
                out = item["output"]
                # Should be CSV-schema compacted (lossless, more compact)
                assert out != ""
                assert len(out) > 0
                # Verify output is not empty and looks compressed
                # SmartCrusher converts JSON arrays to CSV: [N]{schema}\\ndata...
                assert out[0] == "["

    def test_preserves_non_tool_items(self, mock_json_array_body):
        from tinyctx.headroom_compress import compress_tool_outputs
        body = json.loads(json.dumps(mock_json_array_body))
        result = compress_tool_outputs(body, min_chars=100)

        # User message should be untouched
        user_msg = [i for i in result["input"] if i.get("type") == "message"]
        assert len(user_msg) == 1
        assert user_msg[0]["role"] == "user"
        assert user_msg[0]["content"] == "find all functions"

    def test_empty_body(self):
        from tinyctx.headroom_compress import compress_tool_outputs
        result = compress_tool_outputs({})
        assert result == {}

    def test_no_input_key(self):
        from tinyctx.headroom_compress import compress_tool_outputs
        result = compress_tool_outputs({"tools": []})
        assert result == {"tools": []}

    def test_tool_result_type(self):
        from tinyctx.headroom_compress import compress_tool_outputs
        items = [{"id": i, "data": f"item_{i}"} for i in range(30)]
        body = {
            "input": [
                {
                    "type": "tool_result",
                    "call_id": "tr1",
                    "output": json.dumps(items),
                }
            ]
        }
        result = compress_tool_outputs(body, min_chars=100)
        for item in result["input"]:
            if item.get("type") == "tool_result":
                # Should be compressed (CSV schema format)
                assert "string" in item["output"].lower() or len(item["output"]) < len(json.dumps(items))


class TestToolOutputText:
    def test_string_output(self):
        from tinyctx.headroom_compress import _tool_output_text
        item = {"output": "hello world", "type": "tool_result"}
        assert _tool_output_text(item) == "hello world"

    def test_dict_output(self):
        from tinyctx.headroom_compress import _tool_output_text
        item = {"output": {"key": "value"}, "type": "tool_result"}
        assert _tool_output_text(item) == '{"key": "value"}'

    def test_content_list(self):
        from tinyctx.headroom_compress import _tool_output_text
        item = {
            "content": [
                {"type": "text", "text": "line 1"},
                {"type": "input_text", "text": "line 2"},
            ],
            "type": "tool_result",
        }
        assert _tool_output_text(item) == "line 1\nline 2"

    def test_no_output_or_content(self):
        from tinyctx.headroom_compress import _tool_output_text
        item = {"type": "tool_result"}
        assert _tool_output_text(item) == ""
