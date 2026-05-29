"""Pending input park/resume state.

Values may include secrets. Public status/snapshot APIs must expose only
field metadata, never submitted values.
"""
from __future__ import annotations

import pytest


def test_create_status_submit_consume_roundtrip():
    from tinyctx import pending_input, session_state

    session_state.reset_all()
    req = pending_input.create_request(
        "conv-1",
        fields=[{"name": "api_key", "type": "password", "label": "API key"}],
        prompt="Need an API key",
        ttl_s=60.0,
    )

    status = pending_input.status(req["request_id"])
    assert status is not None
    assert status["conv_sid"] == "conv-1"
    assert status["submitted"] is False
    assert status["fields"][0]["name"] == "api_key"
    assert "value" not in status["fields"][0]

    submitted = pending_input.submit(req["request_id"], {"api_key": "sk-secret"})
    assert submitted is not None
    assert submitted["submitted"] is True
    assert "sk-secret" not in repr(submitted)

    consumed = pending_input.consume_submitted("conv-1")
    assert consumed is not None
    assert consumed["values"] == {"api_key": "sk-secret"}
    assert pending_input.status(req["request_id"]) is None


def test_submit_rejects_empty_or_unknown_values_without_hiding_request():
    from tinyctx import pending_input, session_state

    session_state.reset_all()
    req = pending_input.create_request(
        "conv-empty",
        fields=[{"name": "api_key", "type": "password"}],
        prompt="Need an API key",
    )

    with pytest.raises(ValueError):
        pending_input.submit(req["request_id"], {})
    with pytest.raises(ValueError):
        pending_input.submit(req["request_id"], {"wrong": "sk-secret"})

    status = pending_input.status(req["request_id"])
    assert status is not None
    assert status["submitted"] is False
    assert pending_input.peek_submitted("conv-empty") is None


def test_submit_requires_required_fields_before_marking_submitted():
    from tinyctx import pending_input, session_state

    session_state.reset_all()
    req = pending_input.create_request(
        "conv-required",
        fields=[
            {"name": "api_key", "type": "password"},
            {"name": "note", "required": False},
        ],
    )

    with pytest.raises(ValueError):
        pending_input.submit(req["request_id"], {"note": "not enough"})

    status = pending_input.status(req["request_id"])
    assert status is not None
    assert status["submitted"] is False

    submitted = pending_input.submit(
        req["request_id"],
        {"api_key": "sk-secret", "note": "optional"},
    )
    assert submitted is not None
    assert submitted["submitted"] is True


def test_pending_input_status_expires_and_clears(monkeypatch):
    from tinyctx import pending_input, session_state

    session_state.reset_all()
    now = [1000.0]
    monkeypatch.setattr(pending_input.time, "time", lambda: now[0])

    req = pending_input.create_request(
        "conv-expire",
        fields=[{"name": "token", "type": "password"}],
        ttl_s=5.0,
    )
    assert pending_input.status(req["request_id"]) is not None

    now[0] = 1006.0
    assert pending_input.status(req["request_id"]) is None
    assert pending_input.consume_submitted("conv-expire") is None


def test_pending_input_snapshot_scrubs_values():
    from tinyctx import pending_input, session_state

    session_state.reset_all()
    req = pending_input.create_request(
        "conv-scrub",
        fields=[{"name": "password", "type": "password"}],
        ttl_s=60.0,
    )
    pending_input.submit(req["request_id"], {"password": "super-secret"})

    snap = pending_input.snapshot()
    assert req["request_id"] in snap
    assert "super-secret" not in repr(snap)
    assert snap[req["request_id"]]["submitted"] is True


def test_inject_submitted_values_appends_synthetic_user_message():
    from tinyctx import pending_input

    body = {"input": [{"role": "user", "content": "continue"}]}
    submitted = {
        "request_id": "pi_1",
        "prompt": "Need API key",
        "values": {"api_key": "sk-secret"},
    }

    new_body, injected = pending_input.inject_submitted_values(body, submitted)
    assert injected is True
    assert new_body is not body
    text = new_body["input"][-1]["content"][0]["text"]
    assert "Need API key" in text
    assert "api_key: sk-secret" in text
