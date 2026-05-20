"""Tool-call frequency tracking — surface which MCP servers + built-in
tools the agent actually uses, by namespace, rolling totals.

Why
───
Live trace 2026-05-10: `~/.codex/config.toml` registered 6 MCP servers
(gitnexus / serena / advisor / taskmaster-ai / context-mode / advisor)
+ codex's built-in agent protocol tools (spawn_agent, shell, etc.).
But tinyctx had no view of which tools the agent actually CALLED — log
recorded `tools_after` count but not names. Result: silent dead-tool
problem. Examples found in the live trace:

  - `spawn_agent` was being trim_tools'd out (advisor never reachable)
  - `mcp__advisor__ask_advisor` had 0 calls (codex 0.128 namespace bug)
  - we couldn't tell whether gitnexus / serena / etc. were actually
    being used or were dead weight in every request

This module mines `body.input` per request for `function_call` items,
classifies each by namespace (`mcp__<server>__<tool>` → server name,
otherwise "builtin"), and accumulates per-(server, tool, session)
counts. Dedup is by `call_id` so the same call mined across multiple
turns isn't double-counted.

Storage
───────
In-memory only. Resets on proxy restart — that's fine for monitoring
(the dashboard is for "what's happening NOW"). For longitudinal
analysis, jsonl event log is the source of truth.

Interaction
───────────
Hot-path safe — `record_from_body(body)` runs once per request body
mining, NEVER raises. The dashboard endpoint reads a snapshot.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any


# ─── module state ──────────────────────────────────────────────────────────


@dataclass
class _Counts:
    """Per-(namespace, tool_name) aggregate."""
    calls: int = 0
    last_seen_ts: float = 0.0


# {(namespace, tool_name): _Counts}
_BY_TOOL: dict[tuple[str, str], _Counts] = defaultdict(_Counts)
# {namespace: total_calls}
_BY_NAMESPACE: dict[str, int] = defaultdict(int)
# Dedup set — bounded so we don't grow unbounded across long sessions.
# call_ids look like c1/c2/.../codex-uuid; ~32 chars each. Cap at 50K
# entries (~2 MB) — far more than any realistic session generates.
_SEEN_CALL_IDS: set[str] = set()
_SEEN_CALL_IDS_MAX = 50_000

# Coarse lock — recording is NOT in the hot path; happens once per
# request body mine, not per token. Acceptable contention.
_LOCK = Lock()


# ─── classification ───────────────────────────────────────────────────────


def _classify_tool(name: str) -> tuple[str, str]:
    """Return (namespace, tool_name).

    - `mcp__<server>__<tool>` → ("mcp:<server>", "<tool>")
    - codex 0.128+ multi-agent: `spawn_agent` / `wait_agent` etc.
        → ("agent_protocol", name)
    - file/exec built-ins: shell / apply_patch / container.exec
        → ("builtin", name)
    - update_plan / TodoWrite (tracker) → ("tracker", name)
    - everything else → ("other", name)
    """
    if not isinstance(name, str) or not name:
        return ("invalid", "")
    if name.startswith("mcp__"):
        # mcp__<server>__<tool> — split on first __ after the prefix
        rest = name[5:]  # drop "mcp__"
        parts = rest.split("__", 1)
        if len(parts) == 2 and parts[0]:
            return (f"mcp:{parts[0]}", parts[1])
        return ("mcp:unknown", rest)
    if name in ("spawn_agent", "wait_agent", "close_agent",
                "resume_agent", "send_input", "request_user_input",
                "report_agent_job_result"):
        return ("agent_protocol", name)
    # codex 0.128+ renamed `shell` → `exec_command`; both still appear
    # in the wild depending on codex.app version. write_stdin / read_*
    # are the long-running-process companion tools added in 0.128.
    # Live trace 2026-05-10: exec_command had 92 calls misclassified
    # as "other" — fix here keeps the dashboard meaningful.
    if name in ("shell", "exec_command", "apply_patch", "container.exec",
                "local_shell", "view_image", "image_view",
                "write_stdin", "read_stdout", "read_stderr",
                "read_thread_terminal", "kill_command"):
        return ("builtin", name)
    if name in ("update_plan", "TodoWrite"):
        return ("tracker", name)
    return ("other", name)


# ─── recording ─────────────────────────────────────────────────────────────


def record_from_body(body: dict[str, Any]) -> int:
    """Mine `body.input` for `function_call` items; accumulate counts
    by classification. Returns the number of NEW (deduped) calls
    recorded this invocation. Never raises.

    Idempotent across calls within the same request body — a call_id
    seen previously is skipped (so re-mining the same body is safe)."""
    if not isinstance(body, dict):
        return 0
    items = body.get("input")
    if not isinstance(items, list):
        return 0

    new_count = 0
    now = time.time()
    with _LOCK:
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("type") != "function_call":
                continue
            call_id = it.get("call_id") or ""
            if not isinstance(call_id, str) or not call_id:
                # Without a call_id we can't dedup — skip rather than
                # over-count.
                continue
            if call_id in _SEEN_CALL_IDS:
                continue
            name = it.get("name", "") or ""
            namespace, tool = _classify_tool(name)
            key = (namespace, tool)
            ent = _BY_TOOL[key]
            ent.calls += 1
            ent.last_seen_ts = now
            _BY_NAMESPACE[namespace] += 1

            _SEEN_CALL_IDS.add(call_id)
            # Bound memory — drop oldest half if we hit the cap.
            # set ordering isn't preserved but acceptable for dedup.
            if len(_SEEN_CALL_IDS) > _SEEN_CALL_IDS_MAX:
                # Just clear; we'd rather under-count than balloon.
                _SEEN_CALL_IDS.clear()
            new_count += 1
    return new_count


# ─── snapshot for dashboard ───────────────────────────────────────────────


def snapshot() -> dict[str, Any]:
    """Return a JSON-serializable summary of accumulated counts.
    Safe for concurrent access — copies under the lock."""
    with _LOCK:
        by_tool = [
            {
                "namespace": ns,
                "tool": tool,
                "calls": e.calls,
                "last_seen_ts": round(e.last_seen_ts, 1),
                "last_seen_age_s": round(time.time() - e.last_seen_ts, 1)
                                    if e.last_seen_ts else None,
            }
            for (ns, tool), e in _BY_TOOL.items()
        ]
        by_ns = dict(_BY_NAMESPACE)
        seen = len(_SEEN_CALL_IDS)
    # Sort: most-called first
    by_tool.sort(key=lambda d: -d["calls"])
    return {
        "by_tool": by_tool,
        "by_namespace": dict(sorted(by_ns.items(),
                                      key=lambda kv: -kv[1])),
        "total_calls": sum(by_ns.values()),
        "distinct_tools": len(_BY_TOOL),
        "deduped_call_ids_tracked": seen,
    }


# ─── dev/test helpers ─────────────────────────────────────────────────────


def reset_state() -> None:
    with _LOCK:
        _BY_TOOL.clear()
        _BY_NAMESPACE.clear()
        _SEEN_CALL_IDS.clear()
