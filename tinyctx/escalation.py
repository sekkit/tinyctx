"""Graduated escalation ladder — prevents binary "fail → jump to dead frontier"
stalls by escalating through REFINE → PIVOT → SEARCH → BLOCKER levels.

Adapted from the codex-autoresearch pivot protocol. Each level changes strategy
before escalating further, giving the system multiple chances to self-correct.

Concepts mapped to tinyctx:
  "discard" → a failed/non-progress turn (non-200, 0 bytes, tool errors)
  "keep"    → a successful turn that resets the failure counter
  "PIVOT"   → force route to frontier (different model = different strategy)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import session_state


# ─── session state namespace ────────────────────────────────────────────────

_NS = "escalation"
_K_FAILURES = "consecutive_failures"
_K_PIVOTS = "pivot_count"
_K_LEVEL = "last_level"

# Survive compaction (counters are independent of context length), but
# reset on session end (new codex thread = fresh counters).
session_state.register_compaction_reset(_NS, [])
session_state.register_session_end_reset(_NS, [_K_FAILURES, _K_PIVOTS, _K_LEVEL])


# ─── public types ───────────────────────────────────────────────────────────


class EscalationLevel(str, Enum):
    NORMAL = "normal"
    REFINE = "refine"
    PIVOT = "pivot"
    SEARCH = "search"
    BLOCKER = "blocker"


@dataclass(frozen=True)
class EscalationResult:
    level: EscalationLevel
    force_route: str | None  # "frontier" | None
    reminder: str | None     # system-reminder text, None at L0


# ─── reminder templates (Chinese locale) ────────────────────────────────────

_REFINE_REMINDER = """\
<system-reminder>
[NOT USER INPUT — tinyctx escalation ladder: REFINE]

This session has accumulated **{failures} consecutive failed or non-progress turns**. Before escalating to a different model, try adjusting your strategy:

1. **换个角度**: 当前方案的问题是什么? 换个不同的切入点重试。
2. **换粒度**: 如果一直在改大块代码,试试更小的增量改动;反之亦然。
3. **换目标文件**: 如果连续改同一个文件失败,试试从不同文件入手。
4. **回看 lessons**: 有没有之前的经验可以借鉴?

Do NOT continue the same approach that has been failing. Make one concrete strategy adjustment and try again.
</system-reminder>"""

_PIVOT_REMINDER = """\
<system-reminder>
[NOT USER INPUT — tinyctx escalation ladder: PIVOT]

**{failures} consecutive failed turns.** The current strategy is not working. This request is being routed to a frontier model for a fundamentally different approach.

- **Abandon**: 明确放弃当前策略,不要微调。
- **Different angle**: 尝试根本不同的方法(结构性改 vs 增量改,移除 vs 添加,不同层级)。
- **Re-read**: 重新审视目标和约束,确认没有在错误的方向上用力。

The frontier model has different strengths — use them.
</system-reminder>"""

_SEARCH_REMINDER = """\
<system-reminder>
[NOT USER INPUT — tinyctx escalation ladder: SEARCH]

**{pivots} strategy pivots without progress.** The current approaches have been exhausted. Before continuing:

1. **搜索外部知识**: 用 web_search 工具搜索当前阻塞点的解决方案。
2. **查阅文档**: 有没有相关的 API 文档、issue、或讨论可以参考?
3. **换个视角**: 把问题描述给 advisor,让它帮你分析根因。

Do NOT attempt another variation of the same approach.
</system-reminder>"""

_BLOCKER_REMINDER = """\
<system-reminder>
[NOT USER INPUT — tinyctx escalation ladder: BLOCKER]

**{pivots} strategy pivots without any progress.** This goal may require human intervention.

1. **总结阻塞点**: 清晰描述当前目标、尝试过的路径、以及每条路径的失败原因。
2. **向用户汇报**: 用一段话说明当前状况,提出一个具体的需要用户决策的问题。
3. **保存进度**: 确保当前的部分成果已被记录,以便后续恢复。

不要继续尝试。停下来,整理清楚,向用户交棒。
</system-reminder>"""


# ─── core logic ─────────────────────────────────────────────────────────────


def evaluate_escalation(
    consecutive_failures: int,
    pivot_count: int,
) -> EscalationResult:
    """Pure function: determine escalation level from counters.

    Thresholds (adapted from autoresearch pivot protocol):
      L0 NORMAL:  failures < 3
      L1 REFINE:  failures >= 3 (strategy adjustment, same backend)
      L2 PIVOT:   failures >= 5 (force frontier, different approach)
      L3 SEARCH:  pivots >= 2 without keep
      L4 BLOCKER: pivots >= 3 without keep
    """
    f = max(0, int(consecutive_failures))
    p = max(0, int(pivot_count))

    # L4: 3+ pivots without any keep → handoff to human
    if p >= 3:
        return EscalationResult(
            level=EscalationLevel.BLOCKER,
            force_route="frontier",
            reminder=_BLOCKER_REMINDER.format(pivots=p),
        )

    # L3: 2+ pivots without keep → suggest web search
    if p >= 2:
        return EscalationResult(
            level=EscalationLevel.SEARCH,
            force_route="frontier",
            reminder=_SEARCH_REMINDER.format(pivots=p),
        )

    # L2: 5+ consecutive failures → force frontier (PIVOT)
    if f >= 5:
        return EscalationResult(
            level=EscalationLevel.PIVOT,
            force_route="frontier",
            reminder=_PIVOT_REMINDER.format(failures=f),
        )

    # L1: 3+ consecutive failures → REFINE (stay local, adjust strategy)
    if f >= 3:
        return EscalationResult(
            level=EscalationLevel.REFINE,
            force_route=None,
            reminder=_REFINE_REMINDER.format(failures=f),
        )

    # L0: normal operation
    return EscalationResult(
        level=EscalationLevel.NORMAL,
        force_route=None,
        reminder=None,
    )


# ─── per-session state ─────────────────────────────────────────────────────


def _read(conv_sid: str, key: str, default: int = 0) -> int:
    val = session_state.get(conv_sid, _NS, key, default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _write(conv_sid: str, key: str, value: int | str) -> None:
    session_state.set(conv_sid, _NS, key, value)


def evaluate_for_session(conv_sid: str) -> EscalationResult | None:
    """Read the current escalation state for `conv_sid` without mutating it.
    Returns None when conv_sid is falsy or no state exists yet."""
    if not conv_sid:
        return None
    failures = _read(conv_sid, _K_FAILURES)
    pivots = _read(conv_sid, _K_PIVOTS)
    return evaluate_escalation(failures, pivots)


def record_outcome(conv_sid: str, *, ok: bool) -> EscalationResult:
    """Record a turn outcome and return the new escalation state.

    Call this after every request completes.  `ok=True` for a successful
    keep (200 + bytes > 0 + no soft completion); `ok=False` for any
    failure or non-progress outcome.

    Side-effects: updates `consecutive_failures`, `pivot_count`, and
    `last_level` in session_state under `conv_sid`.
    """
    if not conv_sid:
        return evaluate_escalation(0, 0)

    if ok:
        _write(conv_sid, _K_FAILURES, 0)
        _write(conv_sid, _K_PIVOTS, 0)
        _write(conv_sid, _K_LEVEL, EscalationLevel.NORMAL.value)
        return evaluate_escalation(0, 0)

    failures = _read(conv_sid, _K_FAILURES) + 1
    pivots = _read(conv_sid, _K_PIVOTS)
    _write(conv_sid, _K_FAILURES, failures)

    result = evaluate_escalation(failures, pivots)

    # When we cross into PIVOT/SEARCH/BLOCKER, increment the pivot counter
    # so the ladder advances on subsequent evaluations.
    if result.level in (
        EscalationLevel.PIVOT,
        EscalationLevel.SEARCH,
        EscalationLevel.BLOCKER,
    ):
        prev_level = session_state.get(conv_sid, _NS, _K_LEVEL, "")
        if prev_level != result.level.value:
            pivots += 1
            _write(conv_sid, _K_PIVOTS, pivots)

    _write(conv_sid, _K_LEVEL, result.level.value)
    return result


def reset_session(conv_sid: str | None = None) -> None:
    """Clear escalation state.  No arg clears all sessions."""
    if conv_sid is None:
        for sid in list(session_state._STATE.keys()):
            session_state._STATE[sid].pop(_NS, None)
        return
    bucket = session_state._STATE.get(conv_sid)
    if bucket is not None:
        bucket.pop(_NS, None)
