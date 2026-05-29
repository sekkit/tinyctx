"""Priority-ordered pre-flight guard pipeline (P4 refactor).

Before P4, pre-flight guards (`empty_response_guard`, `stuck_loop`,
`synthetic_continue`, `soft_completion`, `plan_persistence`) were called
inline from `tinyctx.proxy.responses` in an implicit, hardcoded order.
Each one's effect (body mutation, route override, flag consumption) was
also inline — adding a new guard meant editing the central handler, and
"why did this guard fire" was scattered across several `_log()` calls.

This module replaces those inline calls with:

- `Guard` — Protocol every guard implements (`name`, `priority`,
  `apply(ctx) -> GuardResult`).
- `GuardContext` — request-scoped inputs guards inspect and mutate
  (body, session keys, turn_count, plus a small mutation accumulator
  for `force_route` and the list of injected reminders).
- `GuardResult` — uniform per-guard contribution (fired? body mutated?
  force_route? reason? additional log fields?).
- `GuardPipeline` — runs guards in priority order (lower first), each
  seeing the cumulative mutations of earlier guards. Exceptions in one
  guard never break the pipeline — they're captured into the guard's
  `GuardResult.reason` so the request can still proceed.

The 5 concrete wrapper guards (`ForceFrontierGuard`,
`BudgetReminderGuard`, `StuckLoopGuard`, `SoftCompletionGate`,
`PlanPersistenceInjector`) preserve EXACT existing behavior — they
only relocate where the call happens. The underlying module functions
(in `empty_response_guard.py` etc.) are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


GUARDRAIL_ACTIONS = {"ok", "repair", "retry", "escalate", "fatal"}


@dataclass
class GuardResult:
    """One guard's contribution to the request shape."""
    guard_name: str
    fired: bool
    reason: str = ""
    body_mutated: bool = False
    force_route: str | None = None  # "frontier" | "local" | None
    additional_log: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureSignal:
    """A normalized guardrail failure signal.

    This is intentionally protocol-neutral: callers may derive it from
    Responses items, SSE chunks, sanitize preflight scans, or translator
    repair attempts, but it never names Chat Completions wire shapes.
    """
    kind: str
    source: str = ""
    severity: int = 1
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any], *, source: str = "") -> "FailureSignal":
        kind = str(value.get("kind") or value.get("type") or "unknown")
        severity = value.get("severity", 1)
        try:
            sev = int(severity)
        except (TypeError, ValueError):
            sev = 1
        detail = {k: v for k, v in value.items()
                  if k not in {"kind", "type", "source", "severity"}}
        return cls(
            kind=kind,
            source=str(value.get("source") or source or ""),
            severity=max(0, sev),
            detail=detail,
        )

    def to_trace(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "severity": self.severity,
            **self.detail,
        }


@dataclass(frozen=True)
class GuardrailDecision:
    """Protocol-neutral decision emitted by guardrail checks.

    Final wire serialization still belongs to tinyctx's Responses/SSE
    emitters; this shape only captures the policy decision.
    """
    action: str
    reason: str = ""
    source: str = ""
    signals: tuple[FailureSignal, ...] = ()
    trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in GUARDRAIL_ACTIONS:
            raise ValueError(f"unknown guardrail action: {self.action!r}")

    @property
    def should_escalate(self) -> bool:
        return self.action == "escalate"

    def to_trace(self) -> dict[str, Any]:
        out = {
            "action": self.action,
            "reason": self.reason,
            "source": self.source,
            "signals": [s.to_trace() for s in self.signals],
        }
        out.update(self.trace)
        return out


class GuardrailErrorTracker:
    """Small per-session consecutive failure counter.

    Borrowed in spirit from forge's ErrorTracker, but scoped to tinyctx
    guardrail signal kinds and intentionally free of retry execution.
    """

    def __init__(self, *, max_consecutive: int = 3):
        self.max_consecutive = max(1, int(max_consecutive))
        self._counts: dict[str, dict[str, int]] = {}

    def record(self, session_id: str, signal_kind: str, *,
               action: str = "retry") -> int:
        sid = session_id or "global"
        kind = signal_kind or "unknown"
        if action == "ok":
            self.reset(sid, kind)
            return 0
        bucket = self._counts.setdefault(sid, {})
        bucket[kind] = bucket.get(kind, 0) + 1
        return bucket[kind]

    def exhausted(self, session_id: str, signal_kind: str) -> bool:
        sid = session_id or "global"
        kind = signal_kind or "unknown"
        return self._counts.get(sid, {}).get(kind, 0) >= self.max_consecutive

    def reset(self, session_id: str | None = None,
              signal_kind: str | None = None) -> None:
        if session_id is None:
            self._counts.clear()
            return
        sid = session_id or "global"
        if signal_kind is None:
            self._counts.pop(sid, None)
            return
        bucket = self._counts.get(sid)
        if bucket is None:
            return
        bucket.pop(signal_kind or "unknown", None)
        if not bucket:
            self._counts.pop(sid, None)

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {sid: dict(counts) for sid, counts in self._counts.items()}


def trace_guard_results(results: list[GuardResult]) -> list[dict[str, Any]]:
    """Return a compact trace-safe view of preflight guard results."""
    return [
        {
            "guard": r.guard_name,
            "fired": r.fired,
            "reason": r.reason,
            "body_mutated": r.body_mutated,
            "force_route": r.force_route,
            **({"log": r.additional_log} if r.additional_log else {}),
        }
        for r in results
    ]


def decision_from_failure_scan(
    scan: dict[str, Any],
    *,
    source: str = "preflight_failure_scan",
    escalate_threshold: int = 2,
) -> GuardrailDecision:
    """Convert sanitize.collect_failure_signals output into a decision."""
    score_raw = scan.get("score", 0)
    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        score = 0
    signals = tuple(
        FailureSignal.from_mapping(s, source=source)
        for s in (scan.get("signals") or [])
        if isinstance(s, dict)
    )
    if score >= escalate_threshold:
        return GuardrailDecision(
            action="escalate",
            reason=f"failure_signal_score={score} >= {escalate_threshold}",
            source=source,
            signals=signals,
            trace={"score": score, "threshold": escalate_threshold},
        )
    return GuardrailDecision(
        action="ok",
        reason=f"failure_signal_score={score}",
        source=source,
        signals=signals,
        trace={"score": score, "threshold": escalate_threshold},
    )


@dataclass
class GuardContext:
    """Inputs guards inspect + mutate.

    `body` is the request body (mutated in-place via reassignment to
    `ctx.body`). `force_route` and `injected_reminders` accumulate the
    cumulative effect of all guards that ran so far; the proxy reads
    them after `pipeline.run()` returns.
    """
    body: dict[str, Any]
    proj_sid: str
    conv_sid: str
    turn_count: int = 0
    is_compaction: bool = False
    forced_by_client_model: bool = False
    # Mutation accumulator
    force_route: str | None = None
    injected_reminders: list[str] = field(default_factory=list)


def _append_synthetic_user_text(
    body: dict[str, Any],
    text: str,
) -> tuple[dict[str, Any], bool]:
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
    if isinstance(messages, list):
        new_messages = list(messages)
        new_messages.append({"role": "user", "content": text})
        out = dict(body)
        out["messages"] = new_messages
        return out, True

    return body, False


@runtime_checkable
class Guard(Protocol):
    """Each guard exposes a name, a priority (lower runs first), and an
    `apply(ctx)` that returns a GuardResult."""
    name: str
    priority: int

    def apply(self, ctx: GuardContext) -> GuardResult: ...


class GuardPipeline:
    """Run a fixed set of guards in priority order. Lower priority runs
    first. Each guard sees ctx mutations from prior guards. A guard's
    exception is captured into its `GuardResult` so subsequent guards
    still run (degrade rather than crash the request)."""

    def __init__(self, guards: list[Guard]):
        self._guards = sorted(guards, key=lambda g: g.priority)

    def run(self, ctx: GuardContext) -> list[GuardResult]:
        results: list[GuardResult] = []
        for g in self._guards:
            try:
                r = g.apply(ctx)
            except Exception as e:  # noqa: BLE001 — degrade, never crash
                r = GuardResult(
                    guard_name=getattr(g, "name", g.__class__.__name__),
                    fired=False,
                    reason=f"exception: {e!r}",
                    additional_log={"exception_type": type(e).__name__},
                )
            results.append(r)
        return results


# ─── concrete wrapper guards ─────────────────────────────────────────────


class ForceFrontierGuard:
    """Consume the one-shot `force_next_to_frontier` flag from
    `empty_response_guard`. When set, the next request is forced to
    frontier (caller reads `ctx.force_route` to apply the route
    override). Tries `conv_sid` first, then falls back to `proj_sid`
    so flags set by mid-stream stall / upstream-error escalation
    (which don't have body access) still trigger.

    Wraps `empty_response_guard.consume_force_frontier`."""

    name = "force_frontier"
    priority = 10  # runs first — its decision dominates routing

    def apply(self, ctx: GuardContext) -> GuardResult:
        from . import empty_response_guard as _erg
        info = _erg.consume_force_frontier(ctx.conv_sid)
        if info is None and ctx.conv_sid != ctx.proj_sid:
            info = _erg.consume_force_frontier(ctx.proj_sid)
        elif info is not None and ctx.conv_sid != ctx.proj_sid:
            # Consuming any flag for this proj_sid clears ALL flags
            # under it (conv_sid-keyed + proj_sid-keyed). Without this,
            # a dangling proj_sid flag would be consumed by a DIFFERENT
            # conversation's next request via the fallback above —
            # force-routing it for no reason. Mirrors proxy.py behavior.
            _erg.reset_state(ctx.proj_sid)
        if info is None:
            return GuardResult(guard_name=self.name, fired=False)
        # Don't force frontier when it's in cooldown — the request
        # would just fail and the flag would re-arm, creating a loop.
        try:
            from . import frontier_health as _fh
            if _fh.is_unreachable():
                return GuardResult(
                    guard_name=self.name, fired=False,
                    reason="frontier cooldown active — deferring force")
        except Exception:  # noqa: BLE001
            pass
        ctx.force_route = "frontier"
        return GuardResult(
            guard_name=self.name,
            fired=True,
            reason=f"empty-response guard: {info.get('reason', '?')[:80]}",
            force_route="frontier",
            additional_log={
                "completion_tokens": info.get("completion_tokens"),
                "finish_reason": info.get("finish_reason"),
            },
        )


class BudgetReminderGuard:
    """Inject a one-shot `<system-reminder>` when synthetic_continue
    tripped its per-conversation injection budget. Skipped on
    compaction or when the client explicitly forced a model.

    Wraps `synthetic_continue.maybe_inject_budget_reminder`."""

    name = "budget_reminder"
    priority = 20

    def __init__(self, max_injections: int = 12):
        self.max_injections = max_injections

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(guard_name=self.name, fired=False,
                                reason="skipped: compaction or forced model")
        from . import synthetic_continue as _syn
        inj_count = _syn.injection_count(ctx.conv_sid)
        if not (inj_count >= self.max_injections and inj_count > 0):
            return GuardResult(guard_name=self.name, fired=False)
        new_body, was_inj = _syn.maybe_inject_budget_reminder(
            ctx.body, ctx.conv_sid, inj_count)
        if not was_inj:
            # Already fired this conversation — flag consumed.
            return GuardResult(guard_name=self.name, fired=False,
                                reason="already fired this conversation")
        ctx.body = new_body
        ctx.injected_reminders.append(self.name)
        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=True,
            reason=f"injection_count={inj_count} >= budget={self.max_injections}",
            additional_log={"injection_count": inj_count},
        )


class EscalationLadderGuard:
    """Graduated escalation ladder (REfiNE → PIVOT → SEARCH → BLOCKER).

    Reads per-session failure/pivot counters from session_state, evaluates
    the current escalation level, and injects the appropriate reminder +
    force_route at each level.  Prevents the binary "fail once → jump to
    dead frontier" stall seen in the Battle City session.

    Priority 25 — runs after budget_reminder (20) but before stuck_loop
    (30), so the ladder can escalate before the old watchdog fires.
    """

    name = "escalation_ladder"
    priority = 25

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction:
            return GuardResult(guard_name=self.name, fired=False,
                                reason="skipped: compaction")
        from . import escalation
        result = escalation.evaluate_for_session(ctx.conv_sid)
        if result is None or result.level == escalation.EscalationLevel.NORMAL:
            return GuardResult(guard_name=self.name, fired=False)

        # Inject reminder into body.input tail (recency-positioned).
        if result.reminder is not None:
            items = ctx.body.get("input")
            if isinstance(items, list):
                new_items = list(items)
                new_items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": result.reminder}],
                })
                out = dict(ctx.body)
                out["input"] = new_items
                ctx.body = out

        # Set force_route at PIVOT and above.
        if result.force_route is not None:
            ctx.force_route = result.force_route

        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=result.reminder is not None,
            force_route=result.force_route,
            reason=f"level={result.level.value}",
            additional_log={
                "escalation_level": result.level.value,
                "force_route": result.force_route,
            },
        )


class StuckLoopGuard:
    """Inject a stuck-loop `<system-reminder>` when turn_count climbs
    past `turn_trigger` without a recent advisor call. Keyed by
    `conv_sid` (so a new codex thread starts with a clean counter);
    advisor grace uses `proj_sid` (so advisor activity in any
    sub-thread quiets nudges across the project).

    Wraps `stuck_loop.maybe_inject_stuck_reminder`."""

    name = "stuck_loop"
    priority = 30

    def __init__(self, turn_trigger: int = 80, turn_gap: int = 50,
                  advisor_grace_s: float = 600.0):
        self.turn_trigger = turn_trigger
        self.turn_gap = turn_gap
        self.advisor_grace_s = advisor_grace_s

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(guard_name=self.name, fired=False,
                                reason="skipped: compaction or forced model")
        from . import stuck_loop
        new_body, was_inj = stuck_loop.maybe_inject_stuck_reminder(
            ctx.body, ctx.conv_sid, ctx.turn_count,
            turn_trigger=self.turn_trigger,
            turn_gap=self.turn_gap,
            advisor_grace_s=self.advisor_grace_s,
            advisor_scope_sid=ctx.proj_sid,
        )
        if not was_inj:
            return GuardResult(guard_name=self.name, fired=False)
        ctx.body = new_body
        ctx.injected_reminders.append(self.name)
        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=True,
            reason=f"stuck at turn={ctx.turn_count}",
            additional_log={"turn_count": ctx.turn_count},
        )


class VerifierGate:
    """Consume the output-quality verifier's force-frontier flag.

    When the verifier scored the previous LOCAL response below the
    quality threshold, force the next request to frontier so the
    higher-quality model can correct the output.

    Priority 35 -- between StuckLoopGuard (30) and SoftCompletionGate
    (40). Quality issues should escalate before soft-completion logic
    injects its own reminders, but after behavioral guards (stuck-loop,
    budget) have had their say.
    """

    name = "verifier_gate"
    priority = 35

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason="skipped: compaction or forced model")
        if ctx.force_route is not None:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason=f"skipped: force_route already={ctx.force_route}")
        from . import verifier
        flag = verifier.get_flag(ctx.proj_sid)
        if flag is None:
            return GuardResult(guard_name=self.name, fired=False)
        action = str(flag.get("action") or "frontier_next")
        if action in ("continue_verify", "local_rewrite"):
            prompt = (
                "[tinyctx verifier — previous local response scored low. "
                "Run verification now, gather concrete execution evidence, "
                "and continue the task without asking the user.]"
            )
            if action == "local_rewrite":
                prompt = (
                    "[tinyctx verifier — previous local response quality scored low. "
                    "Rewrite or correct the answer locally, then verify the result.]"
                )
            new_body, injected = _append_synthetic_user_text(ctx.body, prompt)
            if not injected:
                return GuardResult(guard_name=self.name, fired=False)
            verifier.consume_flag(ctx.proj_sid)
            ctx.body = new_body
            ctx.injected_reminders.append(self.name)
            return GuardResult(
                guard_name=self.name,
                fired=True,
                body_mutated=True,
                reason=f"verifier action={action}: {flag.get('action_reason', '')}",
                additional_log={
                    "total": flag.get("total"),
                    "reason": flag.get("reason"),
                    "action": action,
                },
            )
        verifier.consume_flag(ctx.proj_sid)
        ctx.force_route = "frontier"
        return GuardResult(
            guard_name=self.name,
            fired=True,
            force_route="frontier",
            reason=f"verifier: total={flag.get('total')}/15 "
                   f"({str(flag.get('reason', '?'))[:60]})",
            additional_log={
                "total": flag.get("total"),
                "reason": flag.get("reason"),
            },
        )


class SoftCompletionGate:
    """Inject an advisor-vet `<system-reminder>` when the previous
    turn ended in a "soft punt to user" pattern (matched by the
    streaming sniffer + classifier). Per user directive: if the agent
    insists on asking the user, route the question through advisor
    first.

    Wraps `soft_completion.maybe_inject_soft_completion_gate`."""

    name = "soft_completion_gate"
    priority = 40

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(guard_name=self.name, fired=False,
                                reason="skipped: compaction or forced model")
        from . import soft_completion
        new_body, was_gated, gate_pattern = (
            soft_completion.maybe_inject_soft_completion_gate(
                ctx.body, ctx.proj_sid))
        if not was_gated:
            return GuardResult(guard_name=self.name, fired=False)
        ctx.body = new_body
        ctx.injected_reminders.append(self.name)
        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=True,
            reason=f"soft-completion pattern: {gate_pattern}",
            additional_log={"pattern": gate_pattern},
        )


class AdvisorContinuationGuard:
    """Inject advisor-derived continuation work as synthetic user input on
    the next request after a successful advisor output was observed in the
    previous turn.

    Priority 46 — after SoftCompletionGate (40) and ChoiceArbiterGuard (45),
    before PlanPersistenceInjector (50).
    """

    name = "advisor_continuation"
    priority = 46

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason="skipped: compaction or forced model")
        try:
            from . import advisor_continuation as _ac
        except ImportError:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason="skipped: advisor_continuation module not available")
        pending = _ac.consume_pending_work(ctx.conv_sid)
        if pending is None and ctx.conv_sid != ctx.proj_sid:
            pending = _ac.consume_pending_work(ctx.proj_sid)
        if pending is None:
            return GuardResult(guard_name=self.name, fired=False)
        new_body, was_inj = _ac.inject_pending_work_into_body(ctx.body, pending)
        if not was_inj:
            return GuardResult(guard_name=self.name, fired=False)
        ctx.body = new_body
        ctx.injected_reminders.append(self.name)
        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=True,
            reason=f"injected advisor continuation: {pending.work_text[:80]}",
            additional_log={"source": pending.source},
        )


class PlanPersistenceInjector:
    """Save the current turn's progress tracker (update_plan /
    TodoWrite) to disk for this cwd, and inject a previously-saved
    plan as a `<persisted-plan>` block on the FIRST turn of a fresh
    codex thread (turn_count==0). Bridges context across thread
    boundaries within the same working directory.

    Wraps `plan_persistence.save_plan` + `load_plan` + `inject_plan`."""

    name = "plan_persistence"
    priority = 50  # runs last — purely additive context injection

    def __init__(self, state_dir: Path, cwd: str,
                  plan_ttl_s: float, session_id: str = ""):
        self.state_dir = state_dir
        self.cwd = cwd or ""
        self.plan_ttl_s = plan_ttl_s
        self.session_id = session_id

    def apply(self, ctx: GuardContext) -> GuardResult:
        from . import plan_persistence as _pp
        saved = False
        injected = False
        plan_now = _pp.extract_plan_text(ctx.body)
        if plan_now:
            saved = _pp.save_plan(self.state_dir, self.cwd, plan_now,
                                    session_id=self.session_id,
                                    turn_count=ctx.turn_count)
        pdata_meta: dict[str, Any] = {}
        if ctx.turn_count == 0:
            pdata = _pp.load_plan(self.state_dir, self.cwd,
                                   ttl_s=self.plan_ttl_s)
            if pdata is not None:
                new_body, was_inj = _pp.inject_plan(ctx.body, pdata)
                if was_inj:
                    ctx.body = new_body
                    injected = True
                    pdata_meta = {
                        "prev_turn_count": pdata.get("turn_count_at_save"),
                        "updated": pdata.get("updated_at_iso"),
                    }
        if not (saved or injected):
            return GuardResult(guard_name=self.name, fired=False)
        bits: list[str] = []
        if saved:
            bits.append("saved")
        if injected:
            bits.append("injected")
        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=injected,
            reason=",".join(bits),
            additional_log={"cwd": self.cwd[:120], **pdata_meta},
        )


class ChoiceArbiterGuard:
    """Inject advisor's choice as synthetic user message when the previous
    turn asked the user to pick between options. The verdict is stored by
    `choice_arbiter.intercept()` during stream-rewrite; this guard consumes
    it pre-flight and injects it into body.input so the model sees the
    decision on the next turn without the user typing.

    Priority 45 — after SoftCompletionGate (40) so the gate reminder
    fires first, but before PlanPersistenceInjector (50) so the injected
    user message is part of the body that plan persistence inspects.
    """

    name = "choice_arbiter"
    priority = 45

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason="skipped: compaction or forced model")

        try:
            from . import choice_arbiter as _ca
        except ImportError:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason="skipped: choice_arbiter module not available")
        verdict_sid = ctx.conv_sid
        verdict = _ca.consume_verdict(verdict_sid)
        if verdict is None and ctx.conv_sid != ctx.proj_sid:
            verdict_sid = ctx.proj_sid
            verdict = _ca.consume_verdict(verdict_sid)
        if verdict is None:
            return GuardResult(guard_name=self.name, fired=False)

        new_body, was_inj = _ca.inject_verdict_into_body(ctx.body, verdict)
        if not was_inj:
            _ca.store_verdict(verdict_sid, verdict)
            return GuardResult(guard_name=self.name, fired=False)

        ctx.body = new_body
        ctx.injected_reminders.append(self.name)
        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=True,
            reason=f"injected advisor choice: {verdict.advisor_choice[:80]}",
            additional_log={
                "question": verdict.question[:120],
                "options": verdict.options,
            },
        )


class PendingInputGuard:
    """Inject submitted dashboard input into the next request."""

    name = "pending_input"
    priority = 47

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason="skipped: compaction or forced model")

        from . import pending_input
        submitted_sid = ctx.conv_sid
        submitted = pending_input.peek_submitted(submitted_sid)
        if submitted is None and ctx.conv_sid != ctx.proj_sid:
            submitted_sid = ctx.proj_sid
            submitted = pending_input.peek_submitted(submitted_sid)
        if submitted is None:
            return GuardResult(guard_name=self.name, fired=False)

        new_body, injected = pending_input.inject_submitted_values(
            ctx.body, submitted)
        if not injected:
            return GuardResult(guard_name=self.name, fired=False)

        pending_input.consume_submitted(submitted_sid)
        ctx.body = new_body
        ctx.injected_reminders.append(self.name)
        return GuardResult(
            guard_name=self.name,
            fired=True,
            body_mutated=True,
            reason=f"pending input supplied: {submitted.get('request_id', '')}",
            additional_log={
                "request_id": submitted.get("request_id"),
                "fields": list((submitted.get("values") or {}).keys()),
            },
        )


class SelfImprovementGuard:
    """Force next request to frontier when self-improvement evaluation
    detected performance degradation (elevated error rate or latency).

    The flag is set by `self_improvement._maybe_evaluate()` in the
    post-stream phase. This guard consumes it pre-flight. Priority 60
    ensures more urgent guards (ForceFrontierGuard=10, EscalationLadder=25)
    take precedence."""

    name = "self_improvement"
    priority = 60

    def apply(self, ctx: GuardContext) -> GuardResult:
        if ctx.is_compaction or ctx.forced_by_client_model:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason="skipped: compaction or forced model")
        if ctx.force_route is not None:
            return GuardResult(
                guard_name=self.name, fired=False,
                reason=f"skipped: force_route already={ctx.force_route}")

        from . import self_improvement as _si
        info = _si.consume_degradation_flag(ctx.proj_sid)
        if info is None:
            return GuardResult(guard_name=self.name, fired=False)

        ctx.force_route = "frontier"
        reasons = info.get("reasons", [])
        return GuardResult(
            guard_name=self.name,
            fired=True,
            force_route="frontier",
            reason=f"performance degraded: {','.join(reasons)}",
            additional_log={
                "reasons": reasons,
                "error_rate": info.get("error_rate"),
                "avg_latency_s": info.get("avg_latency_s"),
            },
        )
