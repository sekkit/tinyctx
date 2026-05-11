"""Conversation-identity fingerprint resolution.

# Why this module exists
#
# `prompt_cache_key` was the original signal for per-conversation state
# scoping (synthetic_continue injection counter, empty_response_guard
# force-frontier flag, stuck_loop per-turn reminder gate). It looked
# perfect: a UUID codex attaches to every request in one thread.
#
# Live trace showed it drifts. OpenAI prompt-cache eligibility regenerates
# the key on body-size growth, tool-list shifts, instruction edits, and
# user re-submits in stuck sagas. Forensics dumps from 2026-05-11 confirm:
# three distinct `prompt_cache_key` values across what was almost certainly
# the same logical conversation (the trailing pristine `</INSTRUCTIONS>`
# block of the first user input had identical hash `43ab65f67e29` across
# all three). When `prompt_cache_key` flips mid-conversation, every
# per-conv state-dict effectively resets and the P2 budget cap can never
# fire.
#
# # Stable signals codex DOES send (verified via ~/.tinyctx/forensics/)
#
# - `client_metadata.x-codex-installation-id` — UUID per codex install,
#   stable for the lifetime of the installation.
# - `model` (e.g. `gpt-5.5`, `tinyctx-frontier`) — stable per request flavor.
# - First input item — codex's `<permissions instructions>` developer
#   message, stable across all turns of one conversation and across
#   genuine pck drift. Differs across sandbox-mode changes.
#
# Combining these three yields a key that's STABLE across pck drift,
# DISTINCT for advisor sub-threads (different requested_model), and
# DISTINCT across sandbox-mode shifts. It does NOT distinguish two
# `/clear` conversations in the same install/model/mode — that's an
# accepted trade-off: coarser-than-needed keys merely cause the
# synthetic_continue budget cap to fire slightly early, which is graceful
# degradation. Finer-than-actual keys (the old pck-only behavior) caused
# the cap to NEVER fire, which is the bug.
"""
from __future__ import annotations

import hashlib
from typing import Any


def _developer_block_text(body: dict[str, Any]) -> str:
    """Extract the first developer-role input message's text content.

    Codex always seeds the conversation with a `<permissions instructions>`
    developer message as the first input item. Its content reflects the
    sandbox mode and codex client profile, both stable for one logical
    conversation. After proactive_compact rewrites or stuck-saga retries
    this item stays in position 0 untouched.

    Returns the empty string if no developer item is found (compaction
    handoff bodies, non-codex callers).
    """
    inp = body.get("input")
    if not isinstance(inp, list):
        return ""
    for item in inp[:3]:  # only scan the head; never deeper
        if not isinstance(item, dict):
            continue
        if item.get("role") != "developer":
            continue
        content = item.get("content")
        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    txt = chunk.get("text")
                    if isinstance(txt, str) and txt:
                        return txt
        elif isinstance(content, str):
            return content
    return ""


def _stable_fingerprint(body: dict[str, Any]) -> str:
    """Compute a stable per-conversation fingerprint from drift-resistant fields.

    Combines three signals codex sends on every request:
      - installation id (per-install UUID)
      - model name (separates advisor sub-threads & local vs frontier)
      - first developer-block text hash (per-conv config snapshot)

    Returns an 8-char hex string when at least one signal is non-empty,
    otherwise the empty string (caller falls back to pck or proj_sid).
    """
    if not isinstance(body, dict):
        return ""
    cm = body.get("client_metadata") or {}
    install = ""
    if isinstance(cm, dict):
        v = cm.get("x-codex-installation-id")
        if isinstance(v, str):
            install = v
    model = body.get("model")
    if not isinstance(model, str):
        model = ""
    dev_text = _developer_block_text(body)
    if not install and not dev_text:
        return ""
    h = hashlib.sha256()
    h.update(install.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(dev_text.encode("utf-8"))
    return h.hexdigest()[:8]


def resolve_conv_key(proj_sid: str, body: dict[str, Any]) -> str:
    """Return a stable conversation-scoped composite key.

    Resolution order:
      1. Stable fingerprint (install_id + model + dev-block hash) when any
         signal is present. STABLE across `prompt_cache_key` drift.
      2. `prompt_cache_key` if signal 1 yielded nothing. Older clients
         missing client_metadata land here.
      3. Bare `proj_sid` (full back-compat for compaction handoffs and
         non-codex callers).
    """
    if not isinstance(body, dict):
        return proj_sid
    fp = _stable_fingerprint(body)
    if fp:
        return f"{proj_sid}:fp:{fp}"
    pck = body.get("prompt_cache_key")
    if isinstance(pck, str) and pck:
        return f"{proj_sid}:{pck}"
    return proj_sid
