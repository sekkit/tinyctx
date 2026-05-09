"""Delta-diff for repeat Read tool results.

When the executor re-reads the same file across multiple turns, the
second-and-later `function_call_output` items in the Responses-API body
carry the full file contents again — typically thousands of stable
bytes the upstream model has already seen verbatim. This module detects
those repeats and replaces later occurrences with a unified diff
against the first occurrence.

Idea borrowed (not code) from alexgreensh/token-optimizer's delta_diff:
"first read keeps full content; subsequent reads ship only the diff".
That repo is PolyForm Noncommercial 1.0.0 — we reimplement, no shared
code.

Cache discipline: this transform mutates prompt-cache prefix bytes
for the rewritten items, so callers MUST gate it via
`sanitize.CacheAwareMutator` exactly like dedup_tool_calls /
purge_failed_tool_inputs. The proxy wires this in proxy.py.

Detection — what counts as a "read":
  1. Tool name match: Read, read_file, view, view_file, fs_read, … —
     covers Claude Code, Cursor, generic codex-side read tools.
  2. shell / exec_command / container.exec where the first non-flag
     command token is in {cat, head, tail, less, more, bat, sed -n} —
     covers codex CLI's default file-read pattern.
  3. MCP tools whose last name segment contains read / view / cat —
     covers user-installed MCP servers like fs-read.

Result rewriting:
  - Identical content on re-read → "[tinyctx: re-read of <path> —
    unchanged since first read]" (~70 chars, ~20 tokens).
  - Diff is small enough to be worth it → header + unified diff.
  - Diff is nearly as big as the original → leave the original alone
    (a re-write doesn't compress; we'd just churn cache).
  - Output looks dominated by an error → leave alone (so the model
    keeps seeing the error verbatim and can debug it).
"""
from __future__ import annotations

import difflib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


_TOOL_RESULT_TYPES = {"function_call_output", "tool_result", "mcp_result"}
_TOOL_CALL_TYPES = {"function_call", "tool_use", "mcp_call"}

_READ_TOOL_NAMES = {
    "Read", "read", "read_file", "view", "view_file", "view-file",
    "fs_read", "file_read", "container.read", "open_file",
}

_SHELL_READ_COMMANDS = {"cat", "head", "tail", "less", "more", "bat"}

_SHELL_TOOL_NAMES = {"shell", "exec_command", "container.exec", "bash"}

# Below this length the rewrite isn't worth the placeholder overhead.
_MIN_BYTES = 400

# If diff payload exceeds (original × budget), keep the original.
_DIFF_BUDGET = 0.85

_DELTA_HEADER_PREFIX = "[tinyctx: re-read of"
_UNCHANGED_TEMPLATE = "[tinyctx: re-read of {path} — unchanged since first read]"


@dataclass(frozen=True)
class _ReadKey:
    kind: str          # "read", "shell-cat", "mcp-read"
    path: str

    def display(self) -> str:
        return self.path or f"<{self.kind}>"


def _extract_args(call_item: dict[str, Any]) -> Any:
    """Return the call's arguments parsed as a Python object."""
    args = call_item.get("arguments")
    if args is None:
        args = call_item.get("input")
    if args is None:
        return None
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:  # noqa: BLE001
            return args
    return args


def _normalize_path(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    p = p.strip()
    if not p:
        return ""
    if p.startswith("~"):
        p = os.path.expanduser(p)
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/")


def _path_from_dict(args: dict[str, Any]) -> str:
    for k in ("path", "file_path", "filename", "file", "target_file"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return _normalize_path(v)
    return ""


def _classify_shell_cmd(cmd: list | str) -> str | None:
    """Return a normalized path if `cmd` is a file-read shell command;
    None otherwise. Handles list-form and string-form invocations."""
    if isinstance(cmd, str):
        tokens = cmd.split()
    elif isinstance(cmd, list):
        tokens = [str(t) for t in cmd]
    else:
        return None
    if not tokens:
        return None

    head = tokens[0].rsplit("/", 1)[-1].lower()

    if head in _SHELL_READ_COMMANDS and len(tokens) >= 2:
        for a in tokens[1:]:
            if a.startswith("-"):
                continue
            return _normalize_path(a)
        return None

    # `sed -n '1,200p' /tmp/foo` is the codex default for line-ranged reads.
    if head == "sed" and "-n" in tokens:
        for a in tokens[1:]:
            if a.startswith("-") or a.startswith("'") or "p'" in a or a.endswith("p"):
                # range expression like '1,200p' — skip
                continue
            if "/" in a or a.endswith((".py", ".ts", ".js", ".md", ".toml")):
                return _normalize_path(a)
    return None


def _classify_read(call_item: dict[str, Any]) -> _ReadKey | None:
    """Classify a function_call as a file read; None if it's not."""
    name = (call_item.get("name") or call_item.get("tool_name") or "").strip()
    args = _extract_args(call_item)

    if name in _READ_TOOL_NAMES:
        path = _path_from_dict(args) if isinstance(args, dict) else ""
        return _ReadKey("read", path) if path else None

    if name.startswith("mcp__"):
        last = name.rsplit("__", 1)[-1].lower()
        if any(k in last for k in ("read", "view", "cat")):
            path = _path_from_dict(args) if isinstance(args, dict) else ""
            return _ReadKey("mcp-read", path) if path else None
        return None

    if name in _SHELL_TOOL_NAMES:
        cmd = None
        if isinstance(args, dict):
            cmd = args.get("command") or args.get("cmd")
        path = _classify_shell_cmd(cmd) if cmd is not None else None
        return _ReadKey("shell-cat", path) if path else None

    return None


def _output_text(item: dict[str, Any]) -> str:
    """Stringify a tool-result's payload regardless of shape."""
    raw = item.get("output") if item.get("output") is not None else item.get("content")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for c in raw:
            if isinstance(c, dict):
                t = c.get("type")
                if t in ("text", "input_text", "output_text"):
                    parts.append(str(c.get("text", "")))
        return "\n".join(parts)
    if isinstance(raw, dict):
        t = raw.get("text") or raw.get("content")
        if isinstance(t, str):
            return t
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(raw)


def _set_output_text(item: dict[str, Any], new_text: str) -> None:
    """Write a plain string back into a tool-result, preserving the
    field name (output vs content) and accommodating list-form payloads."""
    target_field = "output" if "output" in item else (
        "content" if "content" in item else "output")
    existing = item.get(target_field)
    if isinstance(existing, list):
        item[target_field] = [{"type": "output_text", "text": new_text}]
    else:
        item[target_field] = new_text


def _looks_like_error(text: str) -> bool:
    """Conservative — return True when the head looks like an error
    payload. We don't want to diff against a `cat: foo: No such file`
    line because the next read might succeed and the diff would be
    nonsense."""
    if not text:
        return True
    head = text[:400].lower()
    markers = (
        "no such file", "permission denied", "is a directory",
        "cannot access", "cannot open", "cannot read",
        "command not found", "exit code: ", "traceback (most recent",
    )
    return any(m in head for m in markers)


def _make_diff(old: str, new: str, *, path: str, n: int = 3) -> str:
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"{path} (first read)",
        tofile=f"{path} (this read)",
        n=n,
        lineterm="",
    )
    return "\n".join(diff)


def collapse_repeated_reads(
    body: dict[str, Any],
    *,
    min_bytes: int = _MIN_BYTES,
    max_diff_budget: float = _DIFF_BUDGET,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace second-and-later same-path Read tool results with a
    unified diff against the first occurrence.

    Returns `(new_body, info)`. `info` keys:
        applied: bool             — at least one rewrite happened
        replacements: int         — items rewritten
        candidates: int           — read-tool results we *considered*
        bytes_saved: int          — sum(len(original) - len(new))
        paths: list[str]          — distinct paths affected (sorted)
        skipped_reasons: dict     — counters for transparency
    """
    info: dict[str, Any] = {
        "applied": False,
        "replacements": 0,
        "candidates": 0,
        "bytes_saved": 0,
        "paths": [],
        "skipped_reasons": {},
    }

    items = body.get("input") or body.get("messages")
    if not isinstance(items, list) or len(items) < 2:
        info["skipped_reasons"]["no_input_array"] = 1
        return body, info

    call_by_id: dict[str, _ReadKey] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") not in _TOOL_CALL_TYPES:
            continue
        cid = it.get("call_id") or it.get("id")
        if not cid:
            continue
        key = _classify_read(it)
        if key is not None and key.path:
            call_by_id[cid] = key

    if not call_by_id:
        info["skipped_reasons"]["no_read_calls"] = 1
        return body, info

    out = deepcopy(body)
    out_items = out.get("input") if isinstance(out.get("input"), list) \
        else out.get("messages")
    if not isinstance(out_items, list):
        return body, info

    seen_first_idx: dict[_ReadKey, int] = {}
    first_text: dict[_ReadKey, str] = {}
    skipped: dict[str, int] = {}

    def _bump(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    affected: set[str] = set()
    bytes_saved = 0
    replacements = 0
    candidates = 0

    for idx, it in enumerate(out_items):
        if not isinstance(it, dict):
            continue
        if it.get("type") not in _TOOL_RESULT_TYPES:
            continue
        cid = it.get("call_id") or it.get("id") or ""
        key = call_by_id.get(cid)
        if key is None:
            continue
        candidates += 1
        text = _output_text(it)
        if len(text) < min_bytes:
            _bump("too_small")
            # Still register first read so subsequent ones diff against
            # it — even if the first read was small, a later large read
            # of the same path is a legitimate baseline-shift; don't
            # penalize the second occurrence by comparing to nothing.
            if key not in seen_first_idx:
                seen_first_idx[key] = idx
                first_text[key] = text
            continue
        if _looks_like_error(text):
            _bump("error_output")
            continue
        if key not in seen_first_idx:
            seen_first_idx[key] = idx
            first_text[key] = text
            continue
        old_text = first_text[key]
        if not old_text:
            # First was empty/too-small; treat THIS as the new baseline.
            seen_first_idx[key] = idx
            first_text[key] = text
            _bump("first_was_too_small")
            continue
        if text == old_text:
            new_text = _UNCHANGED_TEMPLATE.format(path=key.display())
        else:
            diff = _make_diff(old_text, text, path=key.display())
            if not diff:
                _bump("empty_diff")
                continue
            header = (f"{_DELTA_HEADER_PREFIX} {key.display()}, "
                      "diff vs. first read; original above]\n")
            new_text = header + diff
            if len(new_text) > int(len(text) * max_diff_budget):
                _bump("diff_too_large")
                continue
        old_len = len(text)
        _set_output_text(it, new_text)
        bytes_saved += old_len - len(new_text)
        replacements += 1
        affected.add(key.display())

    info["applied"] = replacements > 0
    info["replacements"] = replacements
    info["candidates"] = candidates
    info["bytes_saved"] = bytes_saved
    info["paths"] = sorted(affected)
    info["skipped_reasons"] = skipped
    return out, info
