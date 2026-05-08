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
import os
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


# Regex matching a markdown-style multiple-choice prompt at the end of a
# message. Conservative: requires (a) a "请选择 / pick / which / choose"
# style cue in the last few lines, AND (b) at least 2 enumerated options
# with `A → ...`, `B. ...`, `1) ...` style markers.
# Tuned to NOT fire on incidental "option A" mentions inside reasoning.
_CHOICE_PROMPT_CUE_RE = re.compile(
    r"(请选择|请选|请决定|哪个|"
    r"which (?:option|one)|pick (?:one|an option)|choose (?:one|an option)|"
    r"shall I|should I|do you want)",
    re.IGNORECASE,
)
_CHOICE_OPTION_LINE_RE = re.compile(
    r"^\s*([A-Da-d1-9])\s*[\.\):\->→]\s*\S+",
    re.MULTILINE,
)


def _detect_text_choice_prompt(text: str) -> dict[str, list[str]] | None:
    """Detect a multi-choice prompt at the END of an assistant message
    (model wrote 'pick A or B' as plain text instead of calling
    `request_user_input`). Returns {"header": ..., "options": [...]} when
    confident, None otherwise.

    Conservative: only fires when (1) a cue phrase appears in the last
    ~600 chars AND (2) at least 2 enumerated option lines (`A → ...`,
    `1) ...`) appear after that cue. Inside reasoning prose where the
    model says "option A would be ..." in passing, no match.
    """
    if not text or len(text) < 15:
        return None
    tail = text[-1500:]  # only inspect tail to avoid mid-message false positives
    cue = _CHOICE_PROMPT_CUE_RE.search(tail)
    if not cue:
        return None
    after_cue = tail[cue.start():]
    options = _CHOICE_OPTION_LINE_RE.findall(after_cue)
    if len(options) < 2:
        return None
    # Extract option lines verbatim (label + body)
    option_texts: list[str] = []
    for m in re.finditer(r"^\s*([A-Da-d1-9])\s*[\.\):\->→]\s*(.+?)$",
                         after_cue, re.MULTILINE):
        label = m.group(1).upper()
        body = m.group(2).strip().rstrip(",;.")
        if body:
            option_texts.append(f"{label}: {body}")
    if len(option_texts) < 2:
        return None
    # Header = the line containing the cue (or the line before options)
    cue_line = after_cue[:after_cue.find("\n")].strip() if "\n" in after_cue else after_cue.strip()
    return {"header": cue_line[:200], "options": option_texts}


def _try_auto_answer_text_choice(text: str) -> str | None:
    """Text-level fallback for `_try_auto_answer_user_input`. When the
    model writes a "pick A or B" prompt as plain text instead of calling
    `request_user_input`, detect the pattern and route to the advisor
    anyway. Returns the assistant text suffix to append (advisor's
    decision) or None when env disabled / no choice detected / advisor
    failed.

    Default behavior is ON; set `TINYCTX_AUTO_USER_INPUT=0` to opt out.
    """
    if os.environ.get("TINYCTX_AUTO_USER_INPUT", "1") == "0":
        return None
    detected = _detect_text_choice_prompt(text)
    if not detected:
        return None
    header = detected["header"]
    opts = detected["options"]
    question = (
        "The executor wrote a multi-choice prompt as plain text instead of "
        "calling request_user_input. Pick the best option. Output format:\n"
        "  Pick: <label> — <one-sentence rationale>\n\n"
        f"Header: {header}\n\nOptions:\n" + "\n".join(f"  - {o}" for o in opts)
    )
    try:
        from tinyctx.advisor import call_advisor
        result = call_advisor(
            question=question,
            context="text-level choice intercept (TINYCTX_AUTO_USER_INPUT=1)",
        )
    except Exception:  # noqa: BLE001
        return None
    if result.get("error") or not result.get("text"):
        return None
    advice = result["text"].strip()
    return f"\n\n[advisor auto-decision — text-choice intercept]\n{advice}\n"


def _try_auto_answer_user_input(arguments_json: str) -> str | None:
    """codex 0.128 emits `request_user_input` (a function tool) when the
    model wants the user to pick an option. By default that bubbles up to
    Codex.app's UI as a clickable choice prompt — blocking the session
    until the user clicks. With `TINYCTX_AUTO_USER_INPUT=1`, this helper
    intercepts the call BEFORE codex emits the UI prompt: it synchronously
    consults the advisor (frontier gpt-5.5), parses the chosen option, and
    returns assistant text to be appended to the model's reply. The caller
    then drops the original function_call so codex never sees it.

    Returns None when:
      - the env switch is off,
      - the arguments JSON is malformed / empty,
      - the advisor call fails (timeout / network) — in which case the
        function_call falls back to the normal user-prompt flow.

    The returned text is plain markdown including the advisor's chosen
    option(s) and a short rationale, prefixed with a clear marker so the
    user can audit auto-decisions in chat history.

    Default behavior is ON; set `TINYCTX_AUTO_USER_INPUT=0` to opt out
    and let the choice bubble up to Codex.app's UI for manual click.
    """
    if os.environ.get("TINYCTX_AUTO_USER_INPUT", "1") == "0":
        return None
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return None
    questions = args.get("questions") or []
    if not questions:
        return None

    # Build a tight advisor prompt from the questions array.
    parts = []
    for i, q in enumerate(questions, 1):
        if not isinstance(q, dict):
            continue
        header = q.get("header") or q.get("label") or "(no header)"
        opts = q.get("options") or []
        opts_str = "\n".join(f"  - {o}" for o in opts) if opts else "  (free text)"
        parts.append(f"Q{i}: {header}\nOptions:\n{opts_str}")
    if not parts:
        return None
    question = (
        "The executor is about to ask the user to pick an option. Pick the "
        "best option for each question below. Output format:\n"
        "  Q1: <chosen option> — <one-sentence rationale>\n"
        "  Q2: <chosen option> — <one-sentence rationale>\n\n"
        "Be decisive. If the question has no clear winner, say so but still "
        "pick the safer / more reversible option.\n\n"
        + "\n\n".join(parts)
    )

    # Lazy-import advisor (it's a sibling module that opens an httpx client
    # to the running tinyctx proxy at /v1/responses with model=tinyctx-frontier;
    # circular imports avoided by deferring to call time).
    try:
        from tinyctx.advisor import call_advisor
        result = call_advisor(
            question=question,
            context="auto_user_input intercept — user enabled "
                    "TINYCTX_AUTO_USER_INPUT, so this routes through advisor "
                    "instead of bubbling up to Codex.app's UI.",
        )
    except Exception:  # noqa: BLE001
        return None
    if result.get("error") or not result.get("text"):
        return None
    advice = result["text"].strip()
    return f"\n\n[advisor auto-decision — TINYCTX_AUTO_USER_INPUT=1]\n{advice}\n"


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


# ─────────────── chat-completions SSE → Responses API SSE ───────────────


@dataclass
class ChatToResponsesTranslator:
    """Translates a chat-completions SSE stream into Responses API SSE that
    codex CLI expects. Handles text content, tool_calls, reasoning_content,
    and finish events."""

    _partial: str = ""
    _started: bool = False
    _seq: int = 0
    _text_buf: str = ""
    _reasoning_buf: str = ""
    _resp_id: str = ""
    _model: str = ""
    _item_id: str = field(default_factory=lambda: "msg_" + uuid.uuid4().hex[:24])
    _reasoning_id: str = field(default_factory=lambda: "rs_" + uuid.uuid4().hex[:24])
    _tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    _emitted_calls: int = 0
    _reasoning_item_emitted: bool = False
    _reasoning_done: bool = False
    _message_item_emitted: bool = False

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        self._partial += chunk.decode("utf-8", errors="replace")
        while True:
            sep = self._partial.find("\n\n")
            if sep == -1:
                break
            raw = self._partial[:sep + 2]
            self._partial = self._partial[sep + 2:]
            yield from self._handle_raw(raw)

    def flush(self) -> Iterator[bytes]:
        if self._partial.strip():
            yield from self._handle_raw(self._partial + "\n\n")
            self._partial = ""
        if self._started:
            yield from self._finish()

    def _handle_raw(self, raw: str) -> Iterator[bytes]:
        for line in raw.strip().split("\n"):
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                yield from self._finish()
                return
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if data.get("object") != "chat.completion.chunk":
                continue
            yield from self._handle_chunk(data)

    @property
    def _msg_output_index(self) -> int:
        """Message output_index: 1 when reasoning present, else 0."""
        return 1 if self._reasoning_buf else 0

    def _handle_chunk(self, data: dict[str, Any]) -> Iterator[bytes]:
        if not self._resp_id:
            self._resp_id = data.get("id") or "resp_" + uuid.uuid4().hex[:12]
        if not self._model:
            self._model = data.get("model") or ""

        if not self._started:
            self._started = True
            yield from self._emit_preamble()

        choices = data.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        # ── reasoning_content (DeepSeek thinking mode) ──
        reasoning = delta.get("reasoning_content")
        if reasoning:
            self._reasoning_buf += reasoning
            if not self._reasoning_item_emitted:
                self._reasoning_item_emitted = True
                yield self._sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"id": self._reasoning_id, "type": "reasoning",
                             "summary": []},
                    "sequence_number": self._next_seq(),
                })

        # ── text content ──
        content = delta.get("content")
        if content:
            # Close reasoning item on first content delta
            if self._reasoning_item_emitted and not self._reasoning_done:
                yield from self._close_reasoning_item()
            # Open message item on first content delta
            if not self._message_item_emitted:
                self._message_item_emitted = True
                yield from self._emit_message_item_start()
            self._text_buf += content
            yield self._sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": self._item_id,
                "output_index": self._msg_output_index,
                "content_index": 0,
                "delta": content,
                "sequence_number": self._next_seq(),
            })

        # ── tool_calls ──
        tc_list = delta.get("tool_calls")
        if tc_list:
            for tc in tc_list:
                idx = tc.get("index", 0)
                if idx not in self._tool_calls:
                    self._tool_calls[idx] = {
                        "id": tc.get("id") or "",
                        "name": "",
                        "arguments": "",
                    }
                entry = self._tool_calls[idx]
                if tc.get("id"):
                    entry["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    entry["name"] = fn["name"]
                if fn.get("arguments"):
                    entry["arguments"] += fn["arguments"]

        if finish:
            yield from self._finish()

    def _emit_preamble(self) -> Iterator[bytes]:
        yield self._sse("response.created", {
            "type": "response.created",
            "response": {
                "id": self._resp_id,
                "object": "response",
                "status": "in_progress",
                "model": self._model,
                "output": [],
            },
            "sequence_number": self._next_seq(),
        })
        yield self._sse("response.in_progress", {
            "type": "response.in_progress",
            "response": {"id": self._resp_id, "status": "in_progress"},
            "sequence_number": self._next_seq(),
        })

    def _emit_message_item_start(self) -> Iterator[bytes]:
        oi = self._msg_output_index
        yield self._sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": oi,
            "item": {
                "id": self._item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
            "sequence_number": self._next_seq(),
        })
        yield self._sse("response.content_part.added", {
            "type": "response.content_part.added",
            "item_id": self._item_id,
            "output_index": oi,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
            "sequence_number": self._next_seq(),
        })

    def _close_reasoning_item(self) -> Iterator[bytes]:
        self._reasoning_done = True
        yield self._sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": self._reasoning_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text",
                              "text": self._reasoning_buf}],
            },
            "sequence_number": self._next_seq(),
        })

    def _finish(self) -> Iterator[bytes]:
        if not self._started:
            return
        oi = self._msg_output_index

        # Close reasoning item if still open
        if self._reasoning_item_emitted and not self._reasoning_done:
            yield from self._close_reasoning_item()

        # Ensure message item is opened (even if no content was received)
        if not self._message_item_emitted:
            self._message_item_emitted = True
            yield from self._emit_message_item_start()

        # Auto-answer `request_user_input` via the advisor (env-gated).
        # When TINYCTX_AUTO_USER_INPUT=1, intercept calls to the
        # `request_user_input` function tool, synchronously ask the advisor
        # to pick, append its choice to the assistant text, and DROP the
        # original function_call so codex never raises the UI prompt.
        # Multiple request_user_input calls in one turn all get redirected.
        # Failures (advisor down, malformed args) fall through to normal flow.
        _to_drop = []
        for _idx, _entry in self._tool_calls.items():
            if _entry.get("name") != "request_user_input":
                continue
            _choice_text = _try_auto_answer_user_input(_entry.get("arguments", ""))
            if _choice_text is None:
                continue  # disabled / failed → leave the function_call in place
            self._text_buf += _choice_text
            _to_drop.append(_idx)
        for _idx in _to_drop:
            del self._tool_calls[_idx]

        # Text-level fallback: even when model writes a "pick A or B"
        # prompt as plain prose (NOT a request_user_input call), still
        # intercept it and route to the advisor. Same env switch.
        # Conservative regex: only fires on enumerated option lists
        # following a cue phrase, in the message tail.
        _text_choice = _try_auto_answer_text_choice(self._text_buf)
        if _text_choice is not None:
            self._text_buf += _text_choice

        yield self._sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": self._item_id,
            "output_index": oi,
            "content_index": 0,
            "text": self._text_buf,
            "sequence_number": self._next_seq(),
        })
        yield self._sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": self._item_id,
            "output_index": oi,
            "content_index": 0,
            "part": {"type": "output_text", "text": self._text_buf},
            "sequence_number": self._next_seq(),
        })
        yield self._sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": oi,
            "item": {
                "id": self._item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": self._text_buf}],
            },
            "sequence_number": self._next_seq(),
        })
        # tool_calls start after the message item
        tc_base = oi + 1
        for idx in sorted(self._tool_calls):
            entry = self._tool_calls[idx]
            self._emitted_calls += 1
            oidx = tc_base + self._emitted_calls - 1
            cid = entry["id"] or "fc_" + uuid.uuid4().hex[:24]
            yield self._sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": oidx,
                "item": {"id": cid, "type": "function_call",
                         "status": "in_progress", "name": entry["name"],
                         "arguments": "", "call_id": cid},
                "sequence_number": self._next_seq(),
            })
            yield self._sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": cid, "output_index": oidx,
                "arguments": entry["arguments"],
                "sequence_number": self._next_seq(),
            })
            yield self._sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": oidx,
                "item": {"id": cid, "type": "function_call",
                         "status": "completed", "name": entry["name"],
                         "arguments": entry["arguments"], "call_id": cid},
                "sequence_number": self._next_seq(),
            })

        # Build final output list for response.completed
        output_items: list[dict[str, Any]] = []
        if self._reasoning_buf:
            output_items.append({
                "id": self._reasoning_id, "type": "reasoning",
                "summary": [{"type": "summary_text",
                              "text": self._reasoning_buf}],
            })
        output_items.append({
            "id": self._item_id, "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": self._text_buf}],
        })
        for idx in sorted(self._tool_calls):
            e = self._tool_calls[idx]
            cid = e["id"] or "fc_" + uuid.uuid4().hex[:24]
            output_items.append({
                "id": cid, "type": "function_call", "status": "completed",
                "name": e["name"], "arguments": e["arguments"], "call_id": cid,
            })
        yield self._sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": self._resp_id,
                "object": "response",
                "status": "completed",
                "model": self._model,
                "output": output_items,
            },
            "sequence_number": self._next_seq(),
        })
        self._started = False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @staticmethod
    def _sse(event: str, payload: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
