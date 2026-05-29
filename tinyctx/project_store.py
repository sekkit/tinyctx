"""Per-project persistent stats store.

Each project (identified by cwd from x-codex-cwd header) gets a JSON
state file at ~/.tinyctx/state/projects/<cwd_hash[:16]>.json.

Updates happen in-memory for speed and are flushed to disk
asynchronously every N requests or M seconds. On startup, existing
project files are loaded so stats survive proxy restarts.

Thread-safe — one lock per cwd hash to allow concurrent project writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
STATE_DIR = TINYCTX_HOME / "state" / "projects"
_FLUSH_INTERVAL_REQUESTS = 10  # flush every N requests per project
_FLUSH_INTERVAL_SECONDS = 30.0  # or every M seconds

# ─────────────────────── global state ─────────────────────────────

# In-memory cache: cwd_hash -> project data dict
_cache: dict[str, dict[str, Any]] = {}
# Per-hash lock
_locks: dict[str, threading.Lock] = {}
_global_lock = threading.Lock()  # for _locks and _cache dict access


def _get_lock(cwd_hash: str) -> "threading.RLock":
    with _global_lock:
        if cwd_hash not in _locks:
            # RLock (reentrant): record() holds this lock and then calls
            # _init_project() on a cache miss, which re-acquires the SAME
            # per-hash lock. A plain Lock self-deadlocks there — and because
            # record() runs synchronously on the proxy's event-loop thread
            # (proxy._record_token_tracker at stream end), that froze the
            # entire proxy. RLock lets the same thread re-enter safely.
            _locks[cwd_hash] = threading.RLock()
        return _locks[cwd_hash]


def _cwd_hash(cwd: str) -> str:
    return hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:16]


def _project_path(cwd_hash: str) -> Path:
    return STATE_DIR / f"{cwd_hash}.json"


def _load_project(cwd_hash: str) -> dict[str, Any] | None:
    """Load a project file from disk. Returns None if not found."""
    path = _project_path(cwd_hash)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_project(cwd_hash: str, data: dict[str, Any]) -> None:
    """Atomically write project data to disk (tmp + rename)."""
    path = _project_path(cwd_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def _init_project(cwd: str, cwd_hash: str) -> dict[str, Any]:
    """Create or load a project data dict."""
    lock = _get_lock(cwd_hash)
    with lock:
        if cwd_hash in _cache:
            return _cache[cwd_hash]
        data = _load_project(cwd_hash)
        now = time.time()
        if data is None:
            data = {
                "cwd": cwd,
                "cwd_hash": cwd_hash,
                "first_seen": now,
                "last_seen": now,
                "token": {
                    "requests": 0,
                    "est_input_tokens": 0,
                    "forwarded_tokens": 0,
                    "saved_tokens": 0,
                    "advisor_requests": 0,
                    "advisor_tokens": 0,
                },
                "by_route": {"local": 0, "frontier": 0},
                "flush_count": 0,
            }
            _save_project(cwd_hash, data)
        _cache[cwd_hash] = data
        return data


def record(
    cwd: str = "",
    est_input_tokens: int = 0,
    forwarded_tokens: int = 0,
    route: str = "",
    is_advisor: bool = False,
) -> None:
    """Record one request's stats for a project. Thread-safe."""
    if not cwd:
        return
    ch = _cwd_hash(cwd)
    lock = _get_lock(ch)
    with lock:
        data = _cache.get(ch) or _init_project(cwd, ch)
        t = data["token"]
        t["requests"] += 1
        t["est_input_tokens"] += est_input_tokens
        t["forwarded_tokens"] += forwarded_tokens
        t["saved_tokens"] += max(0, est_input_tokens - forwarded_tokens)
        if is_advisor:
            t["advisor_requests"] += 1
            t["advisor_tokens"] += est_input_tokens
        if route in data["by_route"]:
            data["by_route"][route] += 1
        data["last_seen"] = time.time()
        data["flush_count"] = data.get("flush_count", 0) + 1

        # Flush periodically
        fc = data["flush_count"]
        if fc > 0 and (fc % _FLUSH_INTERVAL_REQUESTS == 0):
            _save_project(ch, data)
            data["flush_count"] = 0


def flush_all() -> None:
    """Force-flush all cached project data to disk. Call on shutdown."""
    with _global_lock:
        for ch, data in list(_cache.items()):
            _save_project(ch, data)


def list_all() -> list[dict[str, Any]]:
    """Return summary list of all known projects, sorted by last_seen desc."""
    # Load any on-disk projects not yet in cache
    if STATE_DIR.is_dir():
        for f in sorted(STATE_DIR.glob("*.json")):
            ch = f.stem
            if ch not in _cache:
                data = _load_project(ch)
                if data:
                    _cache[ch] = data

    with _global_lock:
        projects = list(_cache.values())

    projects.sort(key=lambda p: p.get("last_seen", 0), reverse=True)
    return projects


def get_project(cwd_hash: str) -> dict[str, Any] | None:
    """Get a single project's data by cwd hash."""
    lock = _get_lock(cwd_hash)
    with lock:
        if cwd_hash in _cache:
            return _cache[cwd_hash]
        data = _load_project(cwd_hash)
        if data:
            _cache[cwd_hash] = data
        return data
