"""In-memory pending-input state for park/resume flows.

Secret values intentionally stay in process memory only. Public status
and snapshot helpers scrub submitted values so dashboard/API responses
and routine traces never expose credentials.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from . import session_state


_NS = "pending_input"
_K_REQUESTS = "requests"


def _now() -> float:
    return time.time()


def _requests(conv_sid: str) -> dict[str, dict[str, Any]]:
    existing = session_state.get(conv_sid, _NS, _K_REQUESTS)
    if isinstance(existing, dict):
        return existing
    created: dict[str, dict[str, Any]] = {}
    session_state.set(conv_sid, _NS, _K_REQUESTS, created)
    return created


def _find(request_id: str) -> tuple[str, dict[str, Any]] | None:
    if not request_id:
        return None
    snap = session_state.snapshot()
    for conv_sid, namespaces in snap.items():
        requests = namespaces.get(_NS, {}).get(_K_REQUESTS)
        if isinstance(requests, dict) and request_id in requests:
            req = requests[request_id]
            if isinstance(req, dict):
                return str(conv_sid), req
    return None


def _is_expired(req: dict[str, Any], now: float | None = None) -> bool:
    ttl_s = float(req.get("ttl_s", 0.0) or 0.0)
    if ttl_s <= 0:
        return False
    return (now if now is not None else _now()) > float(req.get("expires_at", 0.0))


def _scrub(req: dict[str, Any]) -> dict[str, Any]:
    fields = []
    for field in req.get("fields", []):
        if not isinstance(field, dict):
            continue
        fields.append({k: v for k, v in field.items() if k != "value"})
    return {
        "request_id": req.get("request_id", ""),
        "conv_sid": req.get("conv_sid", ""),
        "prompt": req.get("prompt", ""),
        "fields": fields,
        "resume_mode": req.get("resume_mode", "park_resume"),
        "cwd": req.get("cwd", ""),
        "created_ts": req.get("created_ts", 0.0),
        "expires_at": req.get("expires_at", 0.0),
        "submitted": bool(req.get("submitted")),
    }


def create_request(
    conv_sid: str,
    *,
    fields: list[dict[str, Any]],
    prompt: str = "",
    ttl_s: float = 300.0,
    resume_mode: str = "park_resume",
    cwd: str = "",
) -> dict[str, Any]:
    request_id = "pi_" + uuid.uuid4().hex[:16]
    created = _now()
    req = {
        "request_id": request_id,
        "conv_sid": conv_sid,
        "prompt": prompt,
        "fields": [
            {
                "name": str(field.get("name", "")),
                "type": str(field.get("type", "text")),
                "label": str(field.get("label", field.get("name", ""))),
                "required": bool(field.get("required", True)),
            }
            for field in fields
            if isinstance(field, dict) and field.get("name")
        ],
        "resume_mode": resume_mode,
        "cwd": str(cwd or ""),
        "created_ts": created,
        "ttl_s": float(ttl_s),
        "expires_at": created + float(ttl_s) if ttl_s > 0 else 0.0,
        "submitted": False,
        "values": {},
    }
    _requests(conv_sid)[request_id] = req
    return _scrub(req)


def build_resume_prompt(submitted: dict[str, Any]) -> str:
    request_id = str(submitted.get("request_id") or "")
    prompt = str(submitted.get("prompt") or "pending input")
    return (
        "[tinyctx pending input resume]\n"
        "The user supplied the requested pending input through the local "
        "dashboard. Continue the interrupted task now. Do not ask again for "
        "that same input; tinyctx will inject the submitted value into this "
        "turn.\n\n"
        f"Request id: {request_id}\n"
        f"Original prompt excerpt: {prompt[:500]}"
    )


def status(request_id: str) -> dict[str, Any] | None:
    found = _find(request_id)
    if found is None:
        return None
    conv_sid, req = found
    if _is_expired(req):
        _requests(conv_sid).pop(request_id, None)
        return None
    return _scrub(req)


def submit(request_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(values, dict):
        raise TypeError("values must be an object")
    found = _find(request_id)
    if found is None:
        return None
    conv_sid, req = found
    if _is_expired(req):
        _requests(conv_sid).pop(request_id, None)
        return None
    provided = {str(k): v for k, v in values.items()}
    cleaned: dict[str, str] = {}
    missing_required: list[str] = []
    for field in req.get("fields", []):
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "")
        if not name:
            continue
        required = bool(field.get("required", True))
        if name not in provided:
            if required:
                missing_required.append(name)
            continue
        value = "" if provided[name] is None else str(provided[name])
        if value == "":
            if required:
                missing_required.append(name)
            continue
        cleaned[name] = value
    if missing_required:
        names = ", ".join(sorted(set(missing_required)))
        raise ValueError(f"missing required fields: {names}")
    if not cleaned:
        raise ValueError("values must include at least one requested field")
    req["values"] = cleaned
    req["submitted"] = True
    req["submitted_ts"] = _now()
    return _scrub(req)


def _submitted(conv_sid: str, *, consume: bool) -> dict[str, Any] | None:
    requests = _requests(conv_sid)
    for request_id, req in list(requests.items()):
        if _is_expired(req):
            requests.pop(request_id, None)
            continue
        if not req.get("submitted"):
            continue
        if consume:
            requests.pop(request_id, None)
        out = _scrub(req)
        out["values"] = dict(req.get("values") or {})
        return out
    return None


def peek_submitted(conv_sid: str) -> dict[str, Any] | None:
    return _submitted(conv_sid, consume=False)


def consume_submitted(conv_sid: str) -> dict[str, Any] | None:
    return _submitted(conv_sid, consume=True)


def snapshot() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    snap = session_state.snapshot()
    for namespaces in snap.values():
        requests = namespaces.get(_NS, {}).get(_K_REQUESTS)
        if not isinstance(requests, dict):
            continue
        for request_id, req in requests.items():
            if isinstance(req, dict) and not _is_expired(req):
                out[str(request_id)] = _scrub(req)
    return out


def inject_submitted_values(
    body: dict[str, Any],
    submitted: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    values = submitted.get("values")
    if not isinstance(values, dict) or not values:
        return body, False
    prompt = submitted.get("prompt") or "Pending input supplied"
    lines = [
        "[tinyctx pending input supplied by the user. Continue the task "
        "using these values; do not ask again for the same input.]",
        "",
        f"Request: {prompt}",
        "",
        "Values:",
    ]
    for key, value in values.items():
        lines.append(f"{key}: {value}")
    text = "\n".join(lines)
    items = body.get("input")
    if isinstance(items, list):
        new_items = list(items)
        new_items.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })
        out = dict(body)
        out["input"] = new_items
        return out, True
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, False
    new_messages = list(messages)
    new_messages.append({
        "role": "user",
        "content": text,
    })
    out = dict(body)
    out["messages"] = new_messages
    return out, True
