"""Unified per-conversation state container.

P1 of the multi-module state-storage consolidation. Modules used to
each carry their own ad-hoc `_X_PER_SESSION: dict[str, ...]` plus
matching set/get/snapshot/reset helpers. This module replaces those
with a single keyed store (`conv_sid` first, `proj_sid` fallback) and
a small typed API.

Storage shape
─────────────
    _STATE[conv_sid][namespace][key] -> value

Namespaces are dotted strings owned by a single module (e.g.
``synthetic_continue``) so keys never collide across modules.

Reset hooks
───────────
Modules register which of their keys reset on a *compaction* boundary
(``register_compaction_reset``). Keys not registered stay across
compactions. ``reset_session_end`` and ``reset_all`` perform broader
wipes for end-of-session and tests respectively.

The store is plain dict access; the proxy is single-event-loop so no
lock is needed.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Optional
import time


# (conv_sid) -> (namespace) -> (key) -> value
_STATE: dict[str, dict[str, dict[str, Any]]] = defaultdict(
    lambda: defaultdict(dict)
)

# namespace -> set of keys that reset_compaction() should clear.
_COMPACTION_RESET_KEYS: dict[str, set[str]] = defaultdict(set)

# namespace -> set of keys that reset_session_end() should clear.
# Defaults to "everything in the namespace" when not registered.
_SESSION_END_RESET_KEYS: dict[str, set[str]] = defaultdict(set)


# ─── registration ────────────────────────────────────────────────────────


def register_compaction_reset(namespace: str, keys: list[str]) -> None:
    """Declare which keys in `namespace` should be cleared by
    `reset_compaction()`. Idempotent — safe to call at module load.
    Keys not registered survive compaction boundaries (e.g. a strategy
    rotation index that is independent of conversation length).
    """
    _COMPACTION_RESET_KEYS[namespace].update(keys)


def register_session_end_reset(namespace: str, keys: list[str]) -> None:
    """Declare which keys reset when the session terminates entirely."""
    _SESSION_END_RESET_KEYS[namespace].update(keys)


# ─── core get/set ────────────────────────────────────────────────────────


def get(conv_sid: Any, namespace: str, key: str,
        default: Any = None) -> Any:
    """Read `key` from `(conv_sid, namespace)`. Returns `default` when
    conv_sid is falsy, namespace is empty, or the key was never set."""
    if not conv_sid:
        return default
    bucket = _STATE.get(conv_sid)
    if bucket is None:
        return default
    ns_bucket = bucket.get(namespace)
    if ns_bucket is None:
        return default
    return ns_bucket.get(key, default)


def set(conv_sid: Any, namespace: str, key: str, value: Any) -> None:
    """Write `key=value` under `(conv_sid, namespace)`. No-op when
    conv_sid is falsy — silently dropping the write is intentional so
    callers don't need to guard against missing conv_sid."""
    if not conv_sid:
        return
    _STATE[conv_sid][namespace][key] = value


def consume(conv_sid: Any, namespace: str, key: str) -> Optional[Any]:
    """Read-and-clear. Returns the previous value, or None if unset.
    Useful for one-shot flags like budget_reminder_fired."""
    if not conv_sid:
        return None
    bucket = _STATE.get(conv_sid)
    if bucket is None:
        return None
    ns_bucket = bucket.get(namespace)
    if ns_bucket is None:
        return None
    return ns_bucket.pop(key, None)


def clear(conv_sid: Any, namespace: str, key: str) -> None:
    """Delete `key` under `(conv_sid, namespace)`. No-op when missing."""
    if not conv_sid:
        return
    bucket = _STATE.get(conv_sid)
    if bucket is None:
        return
    ns_bucket = bucket.get(namespace)
    if ns_bucket is None:
        return
    ns_bucket.pop(key, None)


# ─── counter ─────────────────────────────────────────────────────────────


def increment(conv_sid: Any, namespace: str, key: str,
              by: int = 1) -> int:
    """Atomically increment the integer at `(conv_sid, namespace, key)`
    by `by` and return the new value. Returns 0 unchanged when
    conv_sid is falsy (caller can detect the no-op)."""
    if not conv_sid:
        return 0
    ns_bucket = _STATE[conv_sid][namespace]
    new_val = int(ns_bucket.get(key, 0)) + by
    ns_bucket[key] = new_val
    return new_val


# ─── timestamp ───────────────────────────────────────────────────────────


def mark_timestamp(conv_sid: Any, namespace: str, key: str) -> None:
    """Stamp a monotonic clock reading under `(conv_sid, namespace, key)`.
    Use `seconds_since` to read elapsed time."""
    if not conv_sid:
        return
    _STATE[conv_sid][namespace][key] = time.monotonic()


def seconds_since(conv_sid: Any, namespace: str,
                  key: str) -> Optional[float]:
    """Elapsed seconds since the last `mark_timestamp` for this key, or
    None when never stamped."""
    ts = get(conv_sid, namespace, key)
    if ts is None:
        return None
    return time.monotonic() - float(ts)


# ─── bounded history ─────────────────────────────────────────────────────


def append_bounded(conv_sid: Any, namespace: str, key: str,
                   value: Any, maxlen: int) -> None:
    """Append to a fixed-length history under `(conv_sid, namespace, key)`.
    Older entries are dropped past `maxlen`."""
    if not conv_sid:
        return
    ns_bucket = _STATE[conv_sid][namespace]
    existing = ns_bucket.get(key)
    if not isinstance(existing, deque) or existing.maxlen != maxlen:
        # New deque OR maxlen changed since last call → rebuild.
        seed = list(existing) if isinstance(existing, (list, deque)) else []
        existing = deque(seed, maxlen=maxlen)
        ns_bucket[key] = existing
    existing.append(value)


def get_history(conv_sid: Any, namespace: str,
                key: str) -> list[Any]:
    """Return the current bounded-history contents as a list. Empty list
    when no history exists yet."""
    h = get(conv_sid, namespace, key)
    if h is None:
        return []
    return list(h)


# ─── reset hooks ─────────────────────────────────────────────────────────


def reset_compaction(conv_sid: Any) -> None:
    """Clear per-namespace compaction-scoped keys for `conv_sid`.
    No-op when conv_sid is falsy."""
    if not conv_sid:
        return
    bucket = _STATE.get(conv_sid)
    if bucket is None:
        return
    for namespace, keys in _COMPACTION_RESET_KEYS.items():
        ns_bucket = bucket.get(namespace)
        if not ns_bucket:
            continue
        for key in keys:
            ns_bucket.pop(key, None)


def reset_session_end(conv_sid: Any) -> None:
    """Clear per-namespace session-end-scoped keys for `conv_sid`.
    Falls back to wiping the whole conv when no per-key whitelist is
    registered for a namespace."""
    if not conv_sid:
        return
    bucket = _STATE.get(conv_sid)
    if bucket is None:
        return
    for namespace in list(bucket.keys()):
        keys = _SESSION_END_RESET_KEYS.get(namespace)
        if keys is None:
            continue
        ns_bucket = bucket[namespace]
        for key in keys:
            ns_bucket.pop(key, None)


def reset_all(conv_sid: Optional[Any] = None) -> None:
    """Wipe state. None argument wipes everything (test-only)."""
    if conv_sid is None:
        _STATE.clear()
        return
    _STATE.pop(conv_sid, None)


# ─── snapshot ────────────────────────────────────────────────────────────


def snapshot(conv_sid: Optional[Any] = None) -> dict[str, Any]:
    """Read-only view of state.

    - With `conv_sid`: returns `{namespace: {key: value, ...}, ...}`,
      or `{}` when the conv has no state.
    - Without `conv_sid`: returns `{conv_sid: {namespace: {key: value}}}`
      for every known conv.

    Deque values are exposed as lists so the result is JSON-friendly.
    """
    def _materialize(ns_bucket: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in ns_bucket.items():
            if isinstance(v, deque):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    if conv_sid is not None:
        bucket = _STATE.get(conv_sid)
        if not bucket:
            return {}
        return {ns: _materialize(vals) for ns, vals in bucket.items()}

    return {
        sid: {ns: _materialize(vals) for ns, vals in bucket.items()}
        for sid, bucket in _STATE.items()
    }


# ─── low-level access (for namespaced introspection / sweeps) ─────────────


def keys_with_prefix(prefix: str) -> list[str]:
    """Return all conv_sids that start with `prefix`. Used by modules
    needing the "compaction handoff body lost prompt_cache_key" sweep —
    they delete every per-conv key prefixed by `proj_sid:`."""
    return [sid for sid in _STATE.keys() if sid.startswith(prefix)]
