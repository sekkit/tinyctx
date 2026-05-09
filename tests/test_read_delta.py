"""Tests for tinyctx.read_delta — repeat-read collapse to unified diff."""
from __future__ import annotations

import json
from copy import deepcopy

from tinyctx.read_delta import (
    _DELTA_HEADER_PREFIX,
    _UNCHANGED_TEMPLATE,
    _classify_read,
    collapse_repeated_reads,
)


# -- big enough that min_bytes default (400) doesn't skip them ---------
_BIG_FILE_V1 = "\n".join(f"line {i}: hello world" for i in range(60))
_BIG_FILE_V2_TINY_DIFF = "\n".join(
    f"line {i}: hello world" if i != 5 else f"line {i}: HELLO WORLD"
    for i in range(60))
_BIG_FILE_V3_FAR_DIFF = "\n".join(f"completely different line {i}"
                                  for i in range(60))


def _shell_call(cmd, call_id):
    return {"type": "function_call", "name": "shell",
            "arguments": json.dumps({"command": cmd}), "call_id": call_id}


def _shell_result(text, call_id):
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def _read_call(path, call_id, name="Read"):
    return {"type": "function_call", "name": name,
            "arguments": json.dumps({"path": path}), "call_id": call_id}


# ─────────────────────── classify_read ────────────────────────────────


def test_classify_read_recognizes_named_read_tool():
    key = _classify_read(_read_call("/tmp/foo.py", "c1", name="Read"))
    assert key is not None
    assert key.kind == "read"
    assert key.path == "/tmp/foo.py"


def test_classify_read_recognizes_read_file_alias():
    key = _classify_read(_read_call("/tmp/foo.py", "c1", name="read_file"))
    assert key is not None
    assert key.kind == "read"


def test_classify_read_normalizes_dot_slash_prefix():
    key = _classify_read(_read_call("./src/foo.py", "c1", name="Read"))
    assert key is not None
    assert key.path == "src/foo.py"


def test_classify_read_recognizes_shell_cat():
    key = _classify_read(_shell_call(["cat", "/tmp/log.txt"], "c1"))
    assert key is not None
    assert key.kind == "shell-cat"
    assert key.path == "/tmp/log.txt"


def test_classify_read_recognizes_shell_head_with_flags():
    key = _classify_read(_shell_call(["head", "-50", "/tmp/log.txt"], "c1"))
    assert key is not None
    assert key.path == "/tmp/log.txt"


def test_classify_read_recognizes_shell_with_absolute_path_in_command():
    key = _classify_read(_shell_call(["/usr/bin/cat", "/tmp/log.txt"], "c1"))
    assert key is not None
    assert key.kind == "shell-cat"


def test_classify_read_ignores_shell_non_read_command():
    assert _classify_read(_shell_call(["ls", "/tmp"], "c1")) is None
    assert _classify_read(_shell_call(["pytest"], "c1")) is None
    assert _classify_read(_shell_call(["grep", "foo", "/tmp/x"], "c1")) is None


def test_classify_read_recognizes_mcp_read_tool():
    call = {"type": "function_call", "name": "mcp__filesystem__read_file",
            "arguments": json.dumps({"path": "/tmp/x"}), "call_id": "c1"}
    key = _classify_read(call)
    assert key is not None
    assert key.kind == "mcp-read"
    assert key.path == "/tmp/x"


def test_classify_read_skips_mcp_non_read_tool():
    call = {"type": "function_call", "name": "mcp__advisor__ask_advisor",
            "arguments": "{}", "call_id": "c1"}
    assert _classify_read(call) is None


def test_classify_read_handles_string_command_form():
    """Some tools pass shell command as one string, not a list."""
    call = {"type": "function_call", "name": "shell",
            "arguments": json.dumps({"command": "cat /tmp/x"}),
            "call_id": "c1"}
    key = _classify_read(call)
    assert key is not None
    assert key.path == "/tmp/x"


def test_classify_read_handles_dict_args_directly():
    """Some tool harnesses don't json-encode arguments."""
    call = {"type": "function_call", "name": "Read",
            "arguments": {"path": "/tmp/foo.py"}, "call_id": "c1"}
    key = _classify_read(call)
    assert key is not None
    assert key.path == "/tmp/foo.py"


# ─────────────────── collapse: identity / no-op cases ─────────────────


def test_collapse_no_op_when_no_input_array():
    body = {"model": "x"}
    out, info = collapse_repeated_reads(body)
    assert out == body
    assert info["applied"] is False


def test_collapse_no_op_when_no_read_calls():
    body = {"input": [
        {"role": "user", "content": "hi"},
        {"type": "function_call", "name": "shell",
         "arguments": json.dumps({"command": ["pytest", "-x"]}),
         "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["applied"] is False
    assert info["replacements"] == 0


def test_collapse_keeps_first_read_intact():
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["replacements"] == 0
    # First read's content untouched
    assert out["input"][1]["output"] == _BIG_FILE_V1


def test_collapse_does_not_mutate_input_body():
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
        _read_call("/tmp/foo.py", "c2"),
        _shell_result(_BIG_FILE_V1, "c2"),
    ]}
    snapshot = deepcopy(body)
    _ = collapse_repeated_reads(body)
    assert body == snapshot, "collapse_repeated_reads must not mutate input"


# ────────────────── collapse: replacement happy paths ─────────────────


def test_collapse_replaces_identical_reread_with_unchanged_marker():
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
        {"role": "assistant", "content": "let me look again"},
        _read_call("/tmp/foo.py", "c2"),
        _shell_result(_BIG_FILE_V1, "c2"),
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["applied"] is True
    assert info["replacements"] == 1
    assert info["bytes_saved"] > 0
    assert info["paths"] == ["/tmp/foo.py"]
    # First result intact
    assert out["input"][1]["output"] == _BIG_FILE_V1
    # Second result rewritten to unchanged marker
    second = out["input"][4]["output"]
    assert "unchanged" in second.lower()
    assert "/tmp/foo.py" in second


def test_collapse_replaces_changed_reread_with_unified_diff():
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
        _read_call("/tmp/foo.py", "c2"),
        _shell_result(_BIG_FILE_V2_TINY_DIFF, "c2"),
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["applied"] is True
    assert info["replacements"] == 1
    second = out["input"][3]["output"]
    assert _DELTA_HEADER_PREFIX in second
    # diff markers
    assert "@@" in second
    assert "-line 5: hello world" in second
    assert "+line 5: HELLO WORLD" in second
    # bytes_saved should be substantial — diff is ~few hundred chars vs ~1.5 KB
    assert info["bytes_saved"] > 500


def test_collapse_keeps_third_reread_diff_against_first_not_second():
    """If the file is read 3 times, the 3rd diff is against the FIRST read,
    not the 2nd. This keeps the diff baseline stable across the session
    so the model sees a consistent reference frame."""
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
        _read_call("/tmp/foo.py", "c2"),
        _shell_result(_BIG_FILE_V2_TINY_DIFF, "c2"),
        _read_call("/tmp/foo.py", "c3"),
        _shell_result(_BIG_FILE_V1, "c3"),  # back to v1
    ]}
    out, info = collapse_repeated_reads(body)
    # Both 2nd and 3rd should be replaced; 3rd should say "unchanged"
    # (vs first read), confirming the baseline is the first read.
    assert info["replacements"] == 2
    third = out["input"][5]["output"]
    assert "unchanged" in third.lower()


def test_collapse_skips_when_diff_would_be_too_large():
    """When 90% of lines change, the diff is bigger than the original
    + headers. Keep the original — no point shipping more bytes."""
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
        _read_call("/tmp/foo.py", "c2"),
        _shell_result(_BIG_FILE_V3_FAR_DIFF, "c2"),
    ]}
    out, info = collapse_repeated_reads(body)
    # Should refuse to shrink — the second read body is unchanged.
    assert out["input"][3]["output"] == _BIG_FILE_V3_FAR_DIFF
    # Reflected in skipped_reasons, not counted as a replacement
    assert info["replacements"] == 0
    assert info["skipped_reasons"].get("diff_too_large") == 1


def test_collapse_skips_small_outputs_below_min_bytes():
    body = {"input": [
        _read_call("/tmp/tiny.txt", "c1"),
        _shell_result("tiny", "c1"),
        _read_call("/tmp/tiny.txt", "c2"),
        _shell_result("tiny", "c2"),
    ]}
    out, info = collapse_repeated_reads(body, min_bytes=400)
    assert info["replacements"] == 0
    # both results unchanged
    assert out["input"][1]["output"] == "tiny"
    assert out["input"][3]["output"] == "tiny"


def test_collapse_skips_error_outputs():
    """Don't diff against an error — the next read might succeed and
    leave a confusing diff. Also: errors are short and load-bearing."""
    error = "cat: /tmp/foo: No such file or directory" * 20  # > 400 chars
    body = {"input": [
        _shell_call(["cat", "/tmp/foo"], "c1"),
        _shell_result(error, "c1"),
        _shell_call(["cat", "/tmp/foo"], "c2"),
        _shell_result(error, "c2"),
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["replacements"] == 0
    assert info["skipped_reasons"].get("error_output", 0) >= 1


def test_collapse_handles_list_form_output():
    """Some codex paths emit output as a list of content items, not a
    plain string. The replacement should preserve that shape."""
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        {"type": "function_call_output", "call_id": "c1",
         "output": [{"type": "output_text", "text": _BIG_FILE_V1}]},
        _read_call("/tmp/foo.py", "c2"),
        {"type": "function_call_output", "call_id": "c2",
         "output": [{"type": "output_text", "text": _BIG_FILE_V1}]},
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["applied"] is True
    second = out["input"][3]["output"]
    # Still list-shaped
    assert isinstance(second, list)
    assert second[0]["type"] == "output_text"
    assert "unchanged" in second[0]["text"].lower()


def test_collapse_handles_mixed_call_types_distinct_paths():
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
        _shell_call(["cat", "/tmp/bar.py"], "c2"),
        _shell_result(_BIG_FILE_V2_TINY_DIFF, "c2"),
        _read_call("/tmp/foo.py", "c3"),
        _shell_result(_BIG_FILE_V1, "c3"),
    ]}
    out, info = collapse_repeated_reads(body)
    # c3 dups c1 (same path) → replaced. c2 is a different path, no dup.
    assert info["replacements"] == 1
    assert info["paths"] == ["/tmp/foo.py"]


def test_collapse_path_normalized_dot_slash_treated_as_same():
    body = {"input": [
        _read_call("./src/foo.py", "c1"),
        _shell_result(_BIG_FILE_V1, "c1"),
        _read_call("src/foo.py", "c2"),
        _shell_result(_BIG_FILE_V1, "c2"),
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["replacements"] == 1


def test_collapse_handles_results_with_content_field():
    """tool_result / mcp_result use `content`, not `output`."""
    body = {"input": [
        _read_call("/tmp/foo.py", "c1"),
        {"type": "tool_result", "call_id": "c1", "content": _BIG_FILE_V1},
        _read_call("/tmp/foo.py", "c2"),
        {"type": "tool_result", "call_id": "c2", "content": _BIG_FILE_V1},
    ]}
    out, info = collapse_repeated_reads(body)
    assert info["applied"] is True
    assert "unchanged" in out["input"][3]["content"].lower()


def test_collapse_returns_info_skipped_reasons_when_empty():
    body = {"input": []}
    out, info = collapse_repeated_reads(body)
    assert info["applied"] is False
    assert info["replacements"] == 0
    assert info["bytes_saved"] == 0
    assert isinstance(info["skipped_reasons"], dict)


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
