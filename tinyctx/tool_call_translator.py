"""Translate model-native tool-call formats into OpenAI Responses-API
structured `function_call` items.

Why this exists: codex CLI 0.125 expects tool calls as structured items in
`response.output[]` with shape:

    {"type": "function_call", "name": "...", "arguments": "{...json...}",
     "call_id": "fc_..."}

But several local models (qwen3-coder family in particular, plus what
LMStudio 0.4.12 produces with the Barubary qwen3.5/3.6 chat template)
emit tool calls as raw XML inside an `output_text` content part:

    <tool_call>
    <function=ls>
    <parameter=path>
    /tmp
    </parameter>
    </function>
    </tool_call>

LMStudio 0.4.12 has no per-model tool-call-parser configuration knob (we
checked: it's hardcoded in `libllm_engine.dylib`). The fix has to live at
the proxy layer.

Two paths:

  rebuild_response()    non-streaming JSON response → mutate the output
                        array in place. Easy, deterministic, used when the
                        client requested non-streaming.

  StreamTranslator      streaming SSE state machine. Buffers `response.
                        output_text.delta` events, watches for complete
                        `<tool_call>...</tool_call>` blocks, emits proper
                        Responses-API events: `response.output_item.added`
                        (function_call kind), then a single
                        `response.function_call_arguments.delta` carrying
                        the JSON args, then `response.function_call_
                        arguments.done`, then closes the message item.
                        Used when the client requested streaming.

The translator is purely additive — if the upstream already emits
structured `function_call` items, we leave them alone.
"""
from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator


# ───────────────────────────── format detection ───────────────────────────


# Permissive XML-ish regex tuned to the Barubary template's emit grammar:
#     <tool_call>
#     <function=NAME>
#     <parameter=KEY>
#     VALUE  (may span lines)
#     </parameter>
#     ... more parameters ...
#     </function>
#     </tool_call>
#
# The model occasionally drops a closing tag or trims newlines, so we accept
# both tight and loose whitespace and fail gracefully (return [] from
# parse_tool_call_block on malformed input).
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*"
    r"<function=([^\s>]+)>\s*"
    r"((?:<parameter=[^\s>]+>.*?</parameter>\s*)*?)"
    r"</function>\s*"
    r"</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_PARTIAL_OPEN_RE = re.compile(r"<tool_call\b", re.IGNORECASE)


def parse_tool_call_block(text: str) -> list[dict[str, Any]]:
    """Find every complete `<tool_call>` block in `text` and return one
    `{"name": str, "arguments": str-json}` per call."""
    out: list[dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        params_blob = m.group(2)
        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(params_blob):
            key = pm.group(1).strip()
            val = pm.group(2)
            # Numeric / boolean / JSON guess: if the value parses as JSON,
            # keep it native; else string. This keeps codex-side schema
            # validators happy when they expect typed args.
            args[key] = _coerce_value(val)
        out.append({
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        })
    return out


def _coerce_value(raw: str) -> Any:
    s = raw.strip()
    if s == "":
        return ""
    if s in ("true", "false"):
        return s == "true"
    if s == "null":
        return None
    try:
        if s[0] in "{[\"" or s[0].isdigit() or s[0] == "-":
            return json.loads(s)
    except (ValueError, json.JSONDecodeError):
        pass
    return raw  # keep verbatim (multi-line text, code, etc.)


def _strip_tool_call_blocks(text: str) -> str:
    """Return `text` with every complete `<tool_call>` block removed."""
    return _TOOL_CALL_RE.sub("", text)


# ───────────────────────────── non-streaming ─────────────────────────────


def rebuild_response(response: dict[str, Any]) -> dict[str, Any]:
    """Take a Responses-API completion JSON and return a copy in which any
    `output_text` content containing `<tool_call>` XML has been replaced by
    structured `function_call` items.

    Idempotent: if no XML is present, returns the body unchanged (same dict).
    """
    out_items = response.get("output")
    if not isinstance(out_items, list):
        return response

    new_items: list[dict[str, Any]] = []
    mutated = False
    for item in out_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            new_items.append(item)
            continue
        # Walk the message's content parts looking for output_text with XML.
        contents = item.get("content")
        if not isinstance(contents, list):
            new_items.append(item)
            continue

        residual_parts: list[dict[str, Any]] = []
        extracted_calls: list[dict[str, Any]] = []
        for part in contents:
            if (isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                    and "<tool_call>" in part["text"]):
                text = part["text"]
                calls = parse_tool_call_block(text)
                if calls:
                    extracted_calls.extend(calls)
                    leftover = _strip_tool_call_blocks(text).strip()
                    if leftover:
                        new_part = dict(part)
                        new_part["text"] = leftover
                        residual_parts.append(new_part)
                    continue
            residual_parts.append(part)

        if not extracted_calls:
            new_items.append(item)
            continue

        mutated = True
        # Keep the (possibly-shortened) message item if it has any leftover
        # text content; otherwise drop it.
        if residual_parts:
            new_msg = dict(item)
            new_msg["content"] = residual_parts
            new_items.append(new_msg)
        # Append one function_call item per extracted call.
        for call in extracted_calls:
            cid = "fc_" + uuid.uuid4().hex[:24]
            new_items.append({
                "id": cid,
                "type": "function_call",
                "status": "completed",
                "name": call["name"],
                "arguments": call["arguments"],
                "call_id": cid,
            })

    if not mutated:
        return response
    new_response = deepcopy(response)
    new_response["output"] = new_items
    return new_response


# ─────────────────────────────── streaming ───────────────────────────────


@dataclass
class StreamTranslator:
    """SSE state machine that buffers `response.output_text.delta` events
    until a complete `<tool_call>...</tool_call>` block arrives, then emits
    proper `function_call` events in its place.

    Usage:
        t = StreamTranslator()
        for raw_event_bytes in t.feed(upstream_chunk):
            yield raw_event_bytes
        for raw_event_bytes in t.flush():
            yield raw_event_bytes
    """

    # Per-message-item state (codex assigns one item_id per assistant message).
    _buffers: dict[str, str] = field(default_factory=dict)
    _seq: int = 100_000           # sequence_number for synthesized events
    _emitted_calls: int = 0       # output_index counter for new function_call items
    _saw_partial: dict[str, bool] = field(default_factory=dict)
    _partial: str = ""            # partial event bytes carried across feed() calls

    # ────────── public API ──────────

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        """Receive a raw SSE byte chunk from upstream; yield zero or more raw
        SSE byte chunks to forward to the client."""
        self._partial += chunk.decode("utf-8", errors="replace")
        # SSE events are separated by a blank line: \n\n.
        while True:
            sep = self._partial.find("\n\n")
            if sep == -1:
                break
            raw = self._partial[: sep + 2]
            self._partial = self._partial[sep + 2:]
            yield from self._handle_event(raw)

    def flush(self) -> Iterator[bytes]:
        """Call when upstream stream ends. Drains any half-buffered text
        deltas as plain output_text (not function_call — too risky to
        synthesize from incomplete data)."""
        # Re-emit any incomplete buffered text as a single delta so the
        # client sees the final words of a non-tool message.
        for item_id, buf in self._buffers.items():
            if buf.strip():
                yield self._build_event(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta",
                     "item_id": item_id, "output_index": 0,
                     "content_index": 0, "delta": buf,
                     "sequence_number": self._next_seq()},
                )
        self._buffers.clear()
        if self._partial:
            # Last event without trailing \n\n. Forward as-is.
            tail = self._partial.encode("utf-8")
            self._partial = ""
            if tail:
                yield tail

    # ────────── per-event dispatch ──────────

    def _handle_event(self, raw_event: str) -> Iterator[bytes]:
        # Parse "event: NAME\ndata: JSON\n\n"
        ev_type, data_obj = _parse_sse_event(raw_event)
        if ev_type is None or data_obj is None:
            # Forward unrecognized event verbatim
            yield raw_event.encode("utf-8")
            return

        if ev_type == "response.output_text.delta":
            item_id = str(data_obj.get("item_id", ""))
            delta = data_obj.get("delta", "")
            if not isinstance(delta, str):
                yield raw_event.encode("utf-8")
                return
            buf = self._buffers.get(item_id, "") + delta
            self._buffers[item_id] = buf

            # Strategy: hold back deltas only if we see a partial
            # `<tool_call` opening. Otherwise stream through.
            if not _PARTIAL_OPEN_RE.search(buf):
                # Safe to flush everything as plain text.
                self._buffers[item_id] = ""
                yield self._build_event(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta",
                     "item_id": item_id, "output_index": 0,
                     "content_index": 0, "delta": buf,
                     "sequence_number": self._next_seq()},
                )
                return

            # We're inside a (potentially) tool_call block. Try parsing.
            calls = parse_tool_call_block(buf)
            if not calls:
                # Block not yet complete — keep buffering, emit nothing.
                return

            # We have ≥1 complete tool_call. Flush:
            #   1. residual text before the calls (if any) as one delta
            #   2. one function_call output_item per extracted call
            #   3. drain remaining buffer (text after last </tool_call>)
            head, mid, tail = _split_around_tool_calls(buf)
            self._buffers[item_id] = tail
            if head.strip():
                yield self._build_event(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta",
                     "item_id": item_id, "output_index": 0,
                     "content_index": 0, "delta": head,
                     "sequence_number": self._next_seq()},
                )
            for call in calls:
                yield from self._emit_function_call(call)
            return

        # All other event types: forward unchanged.
        yield raw_event.encode("utf-8")

    # ────────── synthesized event builders ──────────

    def _emit_function_call(self, call: dict[str, Any]) -> Iterator[bytes]:
        self._emitted_calls += 1
        oidx = self._emitted_calls + 1   # +1 to stay clear of the message item
        cid = "fc_" + uuid.uuid4().hex[:24]
        yield self._build_event(
            "response.output_item.added",
            {"type": "response.output_item.added",
             "output_index": oidx,
             "item": {"id": cid, "type": "function_call",
                      "status": "in_progress",
                      "name": call["name"],
                      "arguments": "",
                      "call_id": cid},
             "sequence_number": self._next_seq()},
        )
        yield self._build_event(
            "response.function_call_arguments.delta",
            {"type": "response.function_call_arguments.delta",
             "item_id": cid, "output_index": oidx,
             "delta": call["arguments"],
             "sequence_number": self._next_seq()},
        )
        yield self._build_event(
            "response.function_call_arguments.done",
            {"type": "response.function_call_arguments.done",
             "item_id": cid, "output_index": oidx,
             "arguments": call["arguments"],
             "sequence_number": self._next_seq()},
        )
        yield self._build_event(
            "response.output_item.done",
            {"type": "response.output_item.done",
             "output_index": oidx,
             "item": {"id": cid, "type": "function_call",
                      "status": "completed",
                      "name": call["name"],
                      "arguments": call["arguments"],
                      "call_id": cid},
             "sequence_number": self._next_seq()},
        )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @staticmethod
    def _build_event(name: str, payload: dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


def _parse_sse_event(raw: str) -> tuple[str | None, Any]:
    ev_name = None
    data_str = None
    for line in raw.split("\n"):
        if line.startswith("event:"):
            ev_name = line[6:].strip()
        elif line.startswith("data:"):
            piece = line[5:].lstrip()
            data_str = piece if data_str is None else data_str + piece
    if data_str is None:
        return ev_name, None
    try:
        return ev_name, json.loads(data_str)
    except json.JSONDecodeError:
        return ev_name, None


def _split_around_tool_calls(buf: str) -> tuple[str, str, str]:
    """Return (head_text, _ignored_, tail_text) where head is everything
    before the first `<tool_call>` and tail is everything after the LAST
    `</tool_call>`. The middle (containing the calls themselves) is dropped
    by the caller — it'll be re-emitted as structured events."""
    first = buf.find("<tool_call>")
    if first == -1:
        return buf, "", ""
    head = buf[:first]
    last_close_match = list(_TOOL_CALL_RE.finditer(buf))
    if not last_close_match:
        return head, buf[first:], ""
    end = last_close_match[-1].end()
    tail = buf[end:]
    return head, buf[first:end], tail
