"""Post-stream analysis: classifier, empty-response guard, error terminator.

P7 of the proxy refactor consolidates the ~300 LOC of post-stream logic
that used to live in `proxy._stream_proxy`'s finally / except blocks
into three focused classes:

  - `PostStreamAnalyzer.analyze(ctx)` - spawns the soft-completion
    classifier as a fire-and-forget background task (preserving the
    pre-refactor non-blocking timing) and runs the empty-response guard.
  - `RelayErrorTerminator` - handles `StallCancelledError` and
    `httpx.HTTPError` raised from the relay: builds the SSE error event,
    sets the force-frontier flag (when streak warrants escalation),
    writes forensics.
  - `ForensicsPolicy` - single chokepoint for all forensics dumps. Maps
    a trigger name to the correct gating flag, computes the dump
    directory from `cfg.log_dir`, and forwards to `forensics.write_*`.
    The list of supported triggers is documented in
    `ForensicsPolicy.SUPPORTED_TRIGGERS`.

The behaviour MUST stay byte-identical to the pre-P7 proxy:
  * Same flag reasons (`empty_response: ...`, `stream_error escalate: ...`)
  * Same forensics filename pattern (`<ts>-<trigger>-<uuid8>.json`)
  * Same firing conditions (status==200 AND not upstream_failed for the
    classifier, etc.)

See tests/test_post_stream.py for the behavioural contract.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from . import empty_response_guard as _erg
from . import forensics as _fx
from . import soft_completion as _sc
from . import stall_watchdog as _stall
from .request_phase import RequestPhase, set_phase as _phase_set


# --- ForensicsPolicy ------------------------------------------------------


class ForensicsPolicy:
    """Centralized forensic-dump dispatcher.

    Today the proxy fires forensics dumps at six+ sites (punt,
    punt_via_stream_rewrite, empty_response, stream_error,
    stall_cancelled_relay, upstream_<status>). Each site had its own
    try/except, its own gating flag, and its own derivation of the
    dump dir. This class collapses that into one entry point so:
      - Trigger -> gating-flag mapping lives in ONE place
      - Dump dir derivation lives in ONE place
      - Adding a new trigger is a one-line addition

    Behaviour preserved: each gate-flag (forensics_enabled,
    forensics_capture_punts, forensics_capture_errors) keeps its
    pre-P7 semantics.
    """

    # Triggers the proxy emits today. Used in tests to detect drift.
    # Note: the bg classifier path writes dumps with trigger
    # `punt_p{NN}` (probability suffix) via `soft_completion.
    # write_punt_forensics`; we treat the bare `punt` token as the
    # family head for that case.
    SUPPORTED_TRIGGERS: tuple[str, ...] = (
        "punt",                       # soft-completion bg classifier verdict
        "punt_via_stream_rewrite",    # stream-rewrite synchronous punt
        "empty_response",             # empty_response_guard tripped
        "stream_error",               # httpx.HTTPError mid-stream
        "stall_cancelled_relay",      # watchdog cancelled producer
        # "upstream_<NNN>"            # any 4xx/5xx status - see is_supported_trigger
    )

    # Error-class triggers gated by forensics_capture_errors
    _ERROR_TRIGGERS = frozenset({
        "stream_error", "stall_cancelled_relay",
    })

    # Punt-class triggers gated by forensics_capture_punts
    _PUNT_TRIGGERS = frozenset({"punt", "punt_via_stream_rewrite"})

    def __init__(self, cfg: Any, log: Callable[..., None]) -> None:
        self._cfg = cfg
        self._log = log

    @staticmethod
    def is_supported_trigger(trigger: str) -> bool:
        if trigger in ForensicsPolicy.SUPPORTED_TRIGGERS:
            return True
        if trigger.startswith("upstream_"):
            return True
        # punt_p{NN} family - written by soft_completion.write_punt_forensics
        if trigger.startswith("punt_p"):
            return True
        return False

    def _gating_passes(self, trigger: str) -> bool:
        if not getattr(self._cfg, "forensics_enabled", False):
            return False
        if trigger in self._ERROR_TRIGGERS or trigger.startswith("upstream_"):
            return bool(getattr(self._cfg, "forensics_capture_errors", False))
        if trigger in self._PUNT_TRIGGERS or trigger.startswith("punt_p"):
            return bool(getattr(self._cfg, "forensics_capture_punts", False))
        # empty_response and any other non-error / non-punt trigger:
        # gated only by forensics_enabled (matches existing proxy behavior).
        return True

    def _dump_dir(self) -> Path:
        return Path(self._cfg.log_dir).parent / "forensics"

    def dump(self,
             *,
             trigger: str,
             proj_sid: str,
             response_buffer: str = "",
             response_headers: dict[str, str] | None = None,
             timing: dict[str, float] | None = None,
             classifier_verdict: dict[str, Any] | None = None,
             extra: dict[str, Any] | None = None) -> Path | None:
        """Write a forensics dump. Returns the written path, or None
        when gated off / write fails. Never raises."""
        if not self._gating_passes(trigger):
            return None
        try:
            path = _fx.write_forensics_dump(
                self._dump_dir(),
                proj_sid,
                trigger=trigger,
                response_buffer=response_buffer,
                response_headers=response_headers,
                timing=timing,
                classifier_verdict=classifier_verdict,
                extra=extra,
                max_dumps=getattr(self._cfg, "forensics_max_dumps", 100),
            )
            if path:
                self._log("forensics_dump_written",
                          session=proj_sid, trigger=trigger,
                          path=str(path), file=path.name)
            return path
        except Exception as e:  # noqa: BLE001
            self._log("forensics_dump_error",
                      session=proj_sid, trigger=trigger, error=str(e))
            return None


# --- PostStreamAnalyzer ---------------------------------------------------


@dataclass
class PostStreamContext:
    """All the per-stream state PostStreamAnalyzer needs. Built by the
    relay caller right before invoking `analyze`."""
    proj_sid: str
    conv_sid: str | None
    erg_key: str
    request_id: str
    body: dict[str, Any]
    cwd: str
    bytes_out: int
    status: int
    upstream_failed: bool
    keepalives_emitted: int
    elapsed: float
    started: float
    url: str
    route: str = ""


class PostStreamAnalyzer:
    """Coordinates the post-stream analysis pipeline. Spawned by the
    proxy's `_stream_proxy` finally block - see `analyze(ctx)`.

    Responsibilities:
      1. Spawn the soft-completion classifier as a background task
         (never blocks return).
      2. Run the empty-response guard synchronously.
      3. Forward forensics dump emission to `ForensicsPolicy`.

    Hooks `exec_resume` and the punt forensics dump remain in the
    spawned bg task to preserve the pre-P7 timing (the classifier and
    its side effects run AFTER the stream has fully returned to codex).
    """

    def __init__(self, cfg: Any, log: Callable[..., None]) -> None:
        self._cfg = cfg
        self._log = log
        self._forensics = ForensicsPolicy(cfg=cfg, log=log)

    async def analyze(self, ctx: PostStreamContext) -> None:
        """Run the full post-stream pipeline. Returns immediately after
        the synchronous parts (empty-response guard + forensics); the
        classifier runs as a fire-and-forget background task."""
        self._maybe_run_self_improvement(ctx)
        self._maybe_spawn_classifier(ctx)
        self._maybe_run_empty_response_guard(ctx)

    # -- self-improvement recording (synchronous) --------------------

    def _maybe_run_self_improvement(self, ctx: PostStreamContext) -> None:
        """Record request metrics for self-improvement analysis.

        Runs synchronously (fast: appends to an in-memory ring buffer
        and conditionally triggers a local aggregation)."""
        if not getattr(self._cfg, "self_improvement_enabled", False):
            return
        try:
            from . import self_improvement as _si
            _si.record_request(
                ctx.proj_sid,
                ctx.conv_sid or ctx.proj_sid,
                route=ctx.route,
                status=ctx.status,
                elapsed_s=ctx.elapsed,
                bytes_out=ctx.bytes_out,
                upstream_failed=ctx.upstream_failed,
                cfg=self._cfg,
            )
        except Exception as e:
            self._log("self_improvement_record_error",
                      session=ctx.proj_sid, error=str(e))

    # -- soft-completion classifier (background) ---------------------

    def _maybe_spawn_classifier(self, ctx: PostStreamContext) -> None:
        cfg = self._cfg
        if not getattr(cfg, "soft_completion_gate_enabled", False):
            return
        if ctx.status != 200 or ctx.bytes_out <= 0 or ctx.upstream_failed:
            return
        _phase_set(ctx.proj_sid, RequestPhase.post_stream_classifying,
                   ctx.request_id)
        try:
            api_key = (os.environ.get(cfg.local.api_key_env)
                       if cfg.local.api_key_env else None)
            buffer_snapshot = _sc._OUTPUT_BUFFER.get(ctx.proj_sid, "")
            body_input = (ctx.body.get("input")
                          if isinstance(ctx.body, dict) else None)
            user_goal_snapshot = _sc.extract_user_goal(body_input)
            tracker_snapshot = _sc.extract_progress_tracker(body_input)
            tool_summary_snapshot = _sc.extract_tool_summary(body_input)
            asyncio.create_task(
                self._bg_classify(
                    ctx=ctx,
                    api_key=api_key,
                    buffer_snapshot=buffer_snapshot,
                    user_goal=user_goal_snapshot,
                    tracker=tracker_snapshot,
                    tool_summary=tool_summary_snapshot,
                ))
        except Exception as e:  # noqa: BLE001
            self._log("soft_completion_classify_spawn_error",
                      session=ctx.proj_sid, error=str(e))

    async def _bg_classify(self, *, ctx: PostStreamContext,
                            api_key: str | None,
                            buffer_snapshot: str,
                            user_goal: str,
                            tracker: str,
                            tool_summary: str) -> None:
        cfg = self._cfg
        self._log("soft_completion_classify_started", session=ctx.proj_sid,
                  buffer_chars_at_spawn=len(buffer_snapshot),
                  user_goal_chars=len(user_goal),
                  tracker_chars=len(tracker),
                  tool_summary=tool_summary[:120])
        try:
            diag = await _sc.classify_at_stream_end_diag(
                ctx.proj_sid,
                local_base_url=cfg.local.base_url,
                local_model=cfg.local.model,
                api_key=api_key,
                timeout_s=cfg.self_classify_timeout_s,
                threshold=cfg.self_classify_threshold,
                raw_buffer=buffer_snapshot,
                user_goal=user_goal,
                progress_tracker=tracker,
                tool_summary=tool_summary,
                force_frontier_threshold=(
                    cfg.soft_completion_auto_force_frontier_threshold
                    if cfg.soft_completion_auto_force_frontier_enabled
                    else 1.01),
                short_text_threshold=cfg.soft_completion_short_text_threshold,
                stop_text_threshold=cfg.soft_completion_stop_text_threshold,
                conv_sid=ctx.conv_sid,
                current_route=ctx.route,
            )
            if diag.result is not None:
                self._log("soft_completion_classified",
                          session=ctx.proj_sid,
                          soft_punt=diag.result.soft_punt,
                          p=diag.result.p,
                          reason=diag.result.reason,
                          extracted_text_chars=diag.extracted_text_chars)
                # PUNT forensics dump
                if (diag.result.soft_punt
                        and diag.result.p >= cfg.forensics_punt_threshold):
                    try:
                        forensics_dir = self._forensics._dump_dir()
                        if (getattr(cfg, "forensics_enabled", False)
                                and getattr(cfg, "forensics_capture_punts",
                                            False)):
                            p = _sc.write_punt_forensics(
                                ctx.proj_sid, forensics_dir, diag.result,
                                diag,
                                max_dumps=cfg.forensics_max_dumps)
                            if p:
                                self._log("forensics_dump_written",
                                          session=ctx.proj_sid,
                                          trigger="punt", path=p)
                    except Exception as fe:  # noqa: BLE001
                        self._log("forensics_dump_error",
                                  session=ctx.proj_sid, error=str(fe))
                # exec_resume poke
                if (getattr(cfg, "exec_resume_enabled", False)
                        and diag.result.soft_punt
                        and diag.result.p >= cfg.exec_resume_min_p
                        and ctx.cwd):
                    try:
                        from . import exec_resume as _xr
                        log_dir = Path(cfg.log_dir).parent / "exec_resume_logs"
                        tiers = list(getattr(cfg, "exec_resume_prompt_tiers",
                                              None) or [])
                        rec = await _xr.poke(
                            cwd=ctx.cwd,
                            prompt=cfg.exec_resume_prompt,
                            prompt_tiers=tiers or None,
                            codex_binary=cfg.exec_resume_codex_binary,
                            sandbox=cfg.exec_resume_sandbox,
                            approval_policy=cfg.exec_resume_approval_policy,
                            cooldown_s=cfg.exec_resume_cooldown_s,
                            max_per_minute=cfg.exec_resume_max_per_minute,
                            timeout_s=cfg.exec_resume_timeout_s,
                            log_dir=log_dir,
                            proj_sid=ctx.proj_sid,
                        )
                        self._log("exec_resume_poke",
                                  session=ctx.proj_sid,
                                  status=rec.status, reason=rec.reason,
                                  pid=rec.pid,
                                  resolved_session_id=rec.session_id,
                                  log_path=rec.log_path,
                                  p=diag.result.p)
                    except Exception as xe:  # noqa: BLE001
                        self._log("exec_resume_poke_error",
                                  session=ctx.proj_sid, error=str(xe))
            elif diag.skipped_reason:
                self._log("soft_completion_classify_skipped",
                          session=ctx.proj_sid,
                          reason=diag.skipped_reason,
                          finish_reason=diag.finish_reason,
                          extracted_text_chars=diag.extracted_text_chars,
                          raw_buffer_chars=diag.raw_buffer_chars,
                          raw_head=diag.raw_buffer_head,
                          raw_tail=diag.raw_buffer_tail)
            elif diag.backend_error:
                self._log("soft_completion_classify_backend_error",
                          session=ctx.proj_sid,
                          error=diag.backend_error,
                          status=diag.backend_status,
                          extracted_text_chars=diag.extracted_text_chars)
            else:
                self._log("soft_completion_classify_parse_failed",
                          session=ctx.proj_sid,
                          status=diag.backend_status,
                          raw_preview=diag.raw_content_preview,
                          extracted_text_chars=diag.extracted_text_chars)
        except Exception as e:  # noqa: BLE001
            self._log("soft_completion_classify_error",
                      session=ctx.proj_sid, error=str(e))

    # -- empty-response guard ---------------------------------------

    def _maybe_run_empty_response_guard(self, ctx: PostStreamContext) -> None:
        cfg = self._cfg
        if not getattr(cfg, "empty_response_guard_enabled", False):
            return
        if ctx.status != 200 or ctx.upstream_failed:
            return
        try:
            buf_for_check = _sc._OUTPUT_BUFFER.get(ctx.proj_sid, "")
            info = _erg.maybe_flag_empty_response(
                ctx.erg_key, buf_for_check,
                min_completion_tokens=cfg.empty_response_min_completion_tokens)
            if info is not None:
                self._log("empty_response_detected", session=ctx.proj_sid,
                          completion_tokens=info.get("completion_tokens"),
                          finish_reason=info.get("finish_reason"),
                          reason=info.get("reason"))
                self._forensics.dump(
                    trigger="empty_response",
                    proj_sid=ctx.proj_sid,
                    response_buffer=buf_for_check,
                    timing={
                        "elapsed_s": ctx.elapsed,
                        "started_at": ctx.started,
                    },
                    extra={
                        "bytes_out": ctx.bytes_out,
                        "keepalives_emitted": ctx.keepalives_emitted,
                        "completion_tokens": info.get("completion_tokens"),
                        "finish_reason": info.get("finish_reason"),
                        "url": ctx.url,
                    },
                )
        except Exception as e:  # noqa: BLE001
            self._log("empty_response_guard_error",
                      session=ctx.proj_sid, error=str(e))


# --- RelayErrorTerminator -------------------------------------------------


@dataclass
class RelayErrorResult:
    """Returned to the relay caller so it can yield the SSE error
    event in the right place and update its bookkeeping locals."""
    error_event: bytes
    upstream_failed: bool
    upstream_failure_msg: str
    status: int


class RelayErrorTerminator:
    """Handles the two error classes the relay can raise:

      - `stall_watchdog.StallCancelledError` (synthetic, pushed from
        producer when watchdog cancels)
      - `httpx.HTTPError` (real network / protocol failures)

    For each, it:
      1. Increments the per-session error streak counter
      2. Builds the SSE `event: error` payload (returned to caller to yield)
      3. Sets the force-frontier flag with the right reason string when
         the streak warrants escalation
      4. Writes forensics dump via `ForensicsPolicy`
    """

    def __init__(self, cfg: Any, log: Callable[..., None],
                 session_error_streak: dict[str, int]) -> None:
        self._cfg = cfg
        self._log = log
        self._streak = session_error_streak
        self._forensics = ForensicsPolicy(cfg=cfg, log=log)

    def on_stall_cancelled(self, err: _stall.StallCancelledError,
                            *, proj_sid: str, conv_sid: str | None,
                            bytes_out: int, started: float,
                            url: str) -> RelayErrorResult:
        self._streak[proj_sid] = self._streak.get(proj_sid, 0) + 1
        self._log("stream_error", session=proj_sid, error=str(err),
                  error_type="StallCancelledError",
                  bytes_yielded=bytes_out,
                  elapsed_silent_s=err.elapsed_silent_s,
                  conv_sid=err.conv_sid)
        event = (
            f"event: error\ndata: "
            f"{json.dumps({'message': str(err), 'type': 'stall_cancelled'})}"
            f"\n\n").encode()
        # Forensics
        self._forensics.dump(
            trigger="stall_cancelled_relay",
            proj_sid=proj_sid,
            response_buffer="",
            timing={"elapsed_s": round(time.time() - started, 3),
                    "elapsed_silent_s": err.elapsed_silent_s},
            extra={"conv_sid": err.conv_sid,
                   "bytes_yielded": bytes_out,
                   "url": url},
        )
        return RelayErrorResult(
            error_event=event,
            upstream_failed=True,
            upstream_failure_msg=f"stall_cancelled: {err!s}",
            status=0,
        )

    def on_http_error(self, err: httpx.HTTPError,
                       *, proj_sid: str, conv_sid: str | None,
                       bytes_out: int, started: float,
                       url: str, erg_key: str,
                       request_id: str = "") -> RelayErrorResult:
        self._streak[proj_sid] = self._streak.get(proj_sid, 0) + 1
        self._log("stream_error", session=proj_sid, error=str(err),
                  error_type=type(err).__name__, bytes_yielded=bytes_out)
        event = (
            f"event: error\ndata: "
            f"{json.dumps({'message': str(err)})}\n\n").encode()

        # Transient error escalation
        is_transient = isinstance(err, (
            httpx.RemoteProtocolError, httpx.ReadTimeout,
            httpx.ReadError, httpx.ConnectError, httpx.WriteError))
        cfg = self._cfg
        if (getattr(cfg, "empty_response_guard_enabled", False)
                and is_transient
                and getattr(cfg, "upstream_retry_enabled", False)):
            try:
                if self._streak[proj_sid] > cfg.upstream_retry_count:
                    _erg.force_next_to_frontier(
                        erg_key,
                        f"stream_error escalate: {type(err).__name__} "
                        f"streak={self._streak[proj_sid]}")
                    _phase_set(proj_sid,
                               RequestPhase.escalated_to_frontier, request_id)
                    self._log("stream_error_escalating_to_frontier",
                              session=proj_sid,
                              streak=self._streak[proj_sid])
                else:
                    _phase_set(proj_sid, RequestPhase.retrying, request_id)
                    self._log("stream_error_will_retry_same_backend",
                              session=proj_sid,
                              streak=self._streak[proj_sid])
            except Exception:  # noqa: BLE001 - phase emission is telemetry, not load-bearing
                # Why: phase tracking is observability. A telemetry
                # failure must never crash the post-stream error path -
                # the actual retry decision is already committed above.
                pass

        # Forensics
        self._forensics.dump(
            trigger="stream_error",
            proj_sid=proj_sid,
            response_buffer="",
            timing={"elapsed_s": round(time.time() - started, 3)},
            extra={"error": str(err)[:1000],
                   "error_type": type(err).__name__,
                   "url": url,
                   "bytes_yielded": bytes_out,
                   "session_error_streak": self._streak[proj_sid]},
        )
        return RelayErrorResult(
            error_event=event,
            upstream_failed=True,
            upstream_failure_msg=f"http error: {err!s}",
            status=0,
        )
