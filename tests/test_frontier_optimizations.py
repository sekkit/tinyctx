"""Frontier-only token-budget optimizations.

Covers:
  - trim_tools_for_frontier: keep only tools used in recent window + essentials
  - proactive_compact_only_on_frontier: skip compaction when routing local
  - frontier_skip_advisor_hint: drop the advisor sub-agent hint on frontier

These exist because the local backend (DeepSeek) has a 1M-context window
and is essentially free per token, while the frontier (gpt-5.5 via
chatgpt.com) charges per token AND has a hard 272k internal ceiling.
The user's directive: "本地的小模型不用太节约 token; 主要是给 frontier
要尽量的优化".
"""
from __future__ import annotations

from tinyctx.sanitize import (
    clear_proactive_cache,
    proactive_compact,
    trim_tools_for_frontier,
)


# --- trim_tools_for_frontier ---

def _make_tool(name: str, *, descr: str = "x" * 200) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": descr,
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }


def test_trim_tools_keeps_essentials_and_recent():
    body = {
        "tools": [
            _make_tool("shell"),
            _make_tool("apply_patch"),
            _make_tool("mcp__playwright__navigate"),
            _make_tool("mcp__playwright__click"),
            _make_tool("mcp__youtube__search"),
            _make_tool("mcp__notion__create_page"),
            _make_tool("mcp__advisor__ask_advisor"),
        ],
        "input": [
            # turn 1: nothing
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "search youtube for cats"}]},
            # turn 2: model called mcp__youtube__search
            {"type": "function_call", "call_id": "c1",
             "name": "mcp__youtube__search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            # turn 3: nothing called
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "thanks"}]},
        ],
    }
    out, info = trim_tools_for_frontier(
        body, recent_window=10,
        essentials=("shell", "apply_patch", "mcp__advisor__ask_advisor"),
    )
    assert info["applied"]
    kept = set(info["kept_names"])
    # essentials always kept
    assert "shell" in kept
    assert "apply_patch" in kept
    assert "mcp__advisor__ask_advisor" in kept
    # recently used kept
    assert "mcp__youtube__search" in kept
    # never-used dropped
    assert "mcp__playwright__navigate" not in kept
    assert "mcp__playwright__click" not in kept
    assert "mcp__notion__create_page" not in kept
    # 4 kept of 7
    assert info["tools_after"] == 4
    assert info["tools_before"] == 7


def test_trim_tools_no_op_on_small_list():
    body = {"tools": [_make_tool("shell"), _make_tool("apply_patch")],
            "input": []}
    out, info = trim_tools_for_frontier(body, essentials=("shell",))
    assert not info["applied"]
    assert info["tools_after"] == info["tools_before"] == 2


def test_trim_tools_no_op_when_no_tools():
    out, info = trim_tools_for_frontier({"input": []})
    assert not info["applied"]
    assert info["tools_before"] == 0


def test_trim_tools_no_op_when_all_in_keep_set():
    body = {
        "tools": [_make_tool(f"tool_{i}") for i in range(8)],
        "input": [
            {"type": "function_call", "call_id": f"c{i}",
             "name": f"tool_{i}", "arguments": "{}"}
            for i in range(8)
        ],
    }
    out, info = trim_tools_for_frontier(body, recent_window=20,
                                        essentials=())
    # all 8 used recently → no drop
    assert not info["applied"]
    assert info["tools_after"] == 8
    assert info["reason"] == "all_tools_in_keep_set"


def test_trim_tools_does_not_mutate_input_body():
    import json
    body = {
        "tools": [_make_tool(f"tool_{i}") for i in range(10)],
        "input": [
            {"type": "function_call", "call_id": "c1",
             "name": "tool_3", "arguments": "{}"},
        ],
    }
    snapshot = json.dumps(body, sort_keys=True)
    trim_tools_for_frontier(body, recent_window=10, essentials=())
    assert json.dumps(body, sort_keys=True) == snapshot


def test_trim_tools_recent_window_respected():
    body = {
        "tools": [_make_tool(f"tool_{i}") for i in range(5)],
        "input": [
            # An OLD call that should fall outside recent_window=2
            {"type": "function_call", "call_id": "c0",
             "name": "tool_0", "arguments": "{}"},
            # 4 message items pushing tool_0 out of window
            {"type": "message", "role": "user", "content": [{"type":"input_text","text":"a"}]},
            {"type": "message", "role": "user", "content": [{"type":"input_text","text":"b"}]},
            # A RECENT call in the window
            {"type": "function_call", "call_id": "c1",
             "name": "tool_3", "arguments": "{}"},
        ],
    }
    out, info = trim_tools_for_frontier(body, recent_window=2,
                                        essentials=())
    assert info["applied"]
    kept = set(info["kept_names"])
    assert "tool_3" in kept
    # tool_0 is outside the recent_window
    assert "tool_0" not in kept


# --- end-to-end: confirm the trim accomplishes a meaningful byte savings ---

def test_trim_tools_bytes_savings_realistic():
    """Sanity: trimming 50 → 5 tools where each tool is ~200-byte description
    yields a measurable shrink. Not a strict assertion on exact bytes —
    just confirms the optimization isn't a no-op for realistic inputs."""
    import json
    body = {
        "tools": [_make_tool(f"tool_{i}", descr="x" * 500) for i in range(50)],
        "input": [
            {"type": "function_call", "call_id": "c1",
             "name": "tool_5", "arguments": "{}"},
        ],
    }
    before = len(json.dumps(body))
    out, info = trim_tools_for_frontier(body, recent_window=10,
                                        essentials=("shell",))
    after = len(json.dumps(out))
    assert info["applied"]
    # Should drop 49 of 50 (kept tool_5; "shell" not in tool list, doesn't add)
    assert info["tools_after"] == 1
    # Substantial size win
    assert after < before / 5, (
        f"trim should shrink by >5x for 50→1 tools "
        f"(before={before}, after={after})"
    )
