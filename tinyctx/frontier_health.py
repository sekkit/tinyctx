"""Frontier backend health tracker.

Tracks whether the frontier backend (GPT-5.5 / Opus-class) is reachable.
When connection failures exceed thresholds, marks frontier as unreachable
and enforces a cooldown period with exponential backoff. Auto-recovers
when the cooldown expires.

Thread-safe — all state mutations hold a lock so concurrent request
threads see a consistent view.
"""

from __future__ import annotations

import threading
import time

# ─────────────────────── global state ─────────────────────────────

_lock = threading.Lock()

_frontier_unreachable: bool = False
_consecutive_failures: int = 0
_last_failure_time: float = 0.0
_last_error: str = ""
_cooldown_until: float = 0.0
_frontier_proxy: str = ""

# Exponential backoff: 30 → 60 → 120 → 300 (max) seconds
_BACKOFF_SECONDS = (30.0, 60.0, 120.0, 300.0)
_MAX_BACKOFF = 300.0


def _cooldown_for_failures(n: int) -> float:
    if n <= 0:
        return 0.0
    idx = min(n - 1, len(_BACKOFF_SECONDS) - 1)
    return _BACKOFF_SECONDS[idx]


def _now() -> float:
    return time.time()


def mark_unreachable(error: str = "", proxy: str = "") -> None:
    """Call when a frontier connection fails. Increments failure count
    and sets an exponentially-growing cooldown."""
    global _frontier_unreachable, _consecutive_failures
    global _last_failure_time, _last_error, _cooldown_until, _frontier_proxy
    with _lock:
        _consecutive_failures += 1
        _last_failure_time = _now()
        _last_error = error
        _frontier_proxy = proxy
        _cooldown_until = _last_failure_time + _cooldown_for_failures(
            _consecutive_failures)
        _frontier_unreachable = True


def mark_reachable() -> None:
    """Call when a frontier connection succeeds. Resets all state."""
    global _frontier_unreachable, _consecutive_failures
    global _last_failure_time, _last_error, _cooldown_until
    with _lock:
        _frontier_unreachable = False
        _consecutive_failures = 0
        _last_failure_time = 0.0
        _last_error = ""
        _cooldown_until = 0.0


def is_unreachable() -> bool:
    """Check if frontier is currently marked unreachable and still in
    cooldown. Returns False if cooldown has expired (auto-recovery)."""
    global _frontier_unreachable
    with _lock:
        if not _frontier_unreachable:
            return False
        if _now() >= _cooldown_until:
            # Cooldown expired — allow one retry. Don't fully reset;
            # mark_reachable() will be called on success.
            _frontier_unreachable = False
            return False
        return True


def snapshot() -> dict:
    """Return a read-only snapshot of current frontier health."""
    with _lock:
        remaining = max(0.0, _cooldown_until - _now())
        return {
            "unreachable": _frontier_unreachable and _now() < _cooldown_until,
            "consecutive_failures": _consecutive_failures,
            "last_failure_time": _last_failure_time,
            "last_error": _last_error,
            "cooldown_remaining_s": round(remaining, 1),
            "proxy": _frontier_proxy,
        }


def reminder_text() -> str:
    """Build a <system-reminder> string describing current frontier status
    for injection into the model's context."""
    snap = snapshot()
    if not snap["unreachable"]:
        return ""

    proxy_info = f"当前代理: {snap['proxy']}" if snap["proxy"] else "当前未配置代理（直连）"
    cooldown = f"冷却剩余: {snap['cooldown_remaining_s']}s"

    return (
        f"<system-reminder> 前沿模型 (GPT-5.5 / Opus-class) 当前不可达。"
        f"连接失败 {snap['consecutive_failures']} 次。"
        f"错误: {snap['last_error']}。"
        f"{proxy_info}。{cooldown}。\n\n"
        f"请告知用户此情况，并询问：\n"
        f"1. 是否需要配置代理？可以设置环境变量 TINYCTX_FRONTIER_PROXY "
        f"或在 ~/.tinyctx/config.toml 的 [frontier] 段中设置 proxy。\n"
        f"2. 是否切换为不使用代理直连？（设置 proxy = \"direct\"）\n"
        f"3. 暂时使用本地模型继续工作？\n"
        f"修改配置后需要重启 tinyctx proxy 才能生效。"
        f" </system-reminder>"
    )
