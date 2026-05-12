"""Tests for tinyctx/post_stream.py — the post-stream analysis module.

P7 of the refactor consolidates the post-stream logic that used to live
inline in `proxy._stream_proxy`'s finally block:

  - PostStreamAnalyzer — runs soft-completion classifier (bg) + empty-
    response guard. Sets the force-frontier flag when appropriate.
  - RelayErrorTerminator — handles StallCancelledError / HTTPError ->
    terminator event emission + force-frontier flag + forensics dump.
  - ForensicsPolicy — centralizes the trigger->dump-path naming so all
    forensic dumps go through one entry point with a documented trigger
    list.

These tests pin the BEHAVIOR (not the implementation) so the refactor
can pull code out of proxy.py without subtly drifting timing/flags.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tinyctx import empty_response_guard as _erg
from tinyctx import forensics as _fx
from tinyctx import post_stream as ps
from tinyctx import soft_completion as _sc


# ─── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_state():
    """Each test starts with clean session state."""
    _sc.reset_state()
    _erg.reset_state()
    _fx.reset_state()
    yield
    _sc.reset_state()
    _erg.reset_state()
    _fx.reset_state()


def _mk_cfg(**overrides: Any):
    """Minimal config object — only the fields PostStreamAnalyzer reads."""
    class _Local:
        base_url = "http://local"
        model = "local"
        api_key_env = None

    class _Cfg:
        soft_completion_gate_enabled = True
        soft_completion_auto_force_frontier_enabled = True
        soft_completion_auto_force_frontier_threshold = 0.85
        soft_completion_short_text_threshold = 50
        soft_completion_stop_text_threshold = 1
        self_classify_threshold = 0.7
        self_classify_timeout_s = 30.0
        empty_response_guard_enabled = True
        empty_response_min_completion_tokens = 5
        forensics_enabled = True
        forensics_capture_punts = True
        forensics_punt_threshold = 0.9
        forensics_capture_errors = True
        forensics_max_dumps = 100
        exec_resume_enabled = False
        exec_resume_min_p = 0.85
        local = _Local()
        log_dir = Path("/tmp/tinyctx-test-logs")

    cfg = _Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ─── PostStreamAnalyzer ───────────────────────────────────────────────────


class TestPostStreamAnalyzerHappyPath:
    @pytest.mark.asyncio
    async def test_normal_completion_sets_no_flag(self, tmp_path: Path):
        """Big completion + soft_punt=False -> no flag, no forensics dump."""
        cfg = _mk_cfg(log_dir=tmp_path)
        # Buffer with a valid usage block and ample completion_tokens
        buffer = (
            'event: response.completed\n'
            'data: {"type":"response.completed","response":{'
            '"output":[{"type":"message","content":[{"text":"All set!"}]}],'
            '"usage":{"completion_tokens":200,"finish_reason":"stop"}}}\n\n'
        )
        _sc._OUTPUT_BUFFER["sid-A"] = buffer
        analyzer = ps.PostStreamAnalyzer(cfg=cfg, log=lambda *a, **k: None)

        # Force classifier to say "not a punt"
        from tinyctx.soft_completion import ClassifyDiag, ClassifyResult
        ok_diag = ClassifyDiag(
            result=ClassifyResult(soft_punt=False, p=0.1, reason="ok"),
        )
        with patch.object(_sc, "classify_at_stream_end_diag",
                          return_value=ok_diag) as m:
            ctx = ps.PostStreamContext(
                proj_sid="sid-A", conv_sid="sid-A", erg_key="sid-A",
                request_id="rid-1",
                body={"input": [{"role": "user", "content": "hi"}]},
                cwd="",
                bytes_out=2000,
                status=200,
                upstream_failed=False,
                keepalives_emitted=0,
                elapsed=1.0,
                started=time.time() - 1.0,
                url="http://upstream",
            )
            await analyzer.analyze(ctx)
            # Allow bg task to settle
            await asyncio.sleep(0.05)
            assert m.called

        # No flag set
        assert _erg.peek_force_frontier("sid-A") is None

    @pytest.mark.asyncio
    async def test_zero_bytes_skips_classifier(self, tmp_path: Path):
        """When bytes_out=0 we MUST NOT spawn the classifier (it would
        try to read an empty buffer and log noise)."""
        cfg = _mk_cfg(log_dir=tmp_path)
        analyzer = ps.PostStreamAnalyzer(cfg=cfg, log=lambda *a, **k: None)
        with patch.object(_sc, "classify_at_stream_end_diag") as m:
            ctx = ps.PostStreamContext(
                proj_sid="sid", conv_sid="sid", erg_key="sid",
                request_id="rid", body={}, cwd="",
                bytes_out=0, status=200, upstream_failed=False,
                keepalives_emitted=0, elapsed=0.1, started=time.time(),
                url="http://x",
            )
            await analyzer.analyze(ctx)
            await asyncio.sleep(0.05)
            assert not m.called

    @pytest.mark.asyncio
    async def test_upstream_failed_skips_classifier(self, tmp_path: Path):
        cfg = _mk_cfg(log_dir=tmp_path)
        analyzer = ps.PostStreamAnalyzer(cfg=cfg, log=lambda *a, **k: None)
        with patch.object(_sc, "classify_at_stream_end_diag") as m:
            ctx = ps.PostStreamContext(
                proj_sid="sid", conv_sid="sid", erg_key="sid",
                request_id="rid", body={}, cwd="",
                bytes_out=1000, status=200, upstream_failed=True,
                keepalives_emitted=0, elapsed=0.1, started=time.time(),
                url="http://x",
            )
            await analyzer.analyze(ctx)
            await asyncio.sleep(0.05)
            assert not m.called


class TestPostStreamAnalyzerEmptyResponse:
    @pytest.mark.asyncio
    async def test_low_completion_tokens_sets_force_frontier(self, tmp_path: Path):
        cfg = _mk_cfg(log_dir=tmp_path)
        # Buffer with very small completion_tokens + finish_reason=stop
        buffer = (
            'event: response.completed\n'
            'data: {"usage":{"completion_tokens":1,"finish_reason":"stop"}}\n\n'
        )
        _sc._OUTPUT_BUFFER["sid-E"] = buffer
        analyzer = ps.PostStreamAnalyzer(cfg=cfg, log=lambda *a, **k: None)
        # Skip the classifier work (return a benign no-punt verdict)
        from tinyctx.soft_completion import ClassifyDiag, ClassifyResult
        ok_diag = ClassifyDiag(
            result=ClassifyResult(soft_punt=False, p=0.1, reason="ok"),
        )
        with patch.object(_sc, "classify_at_stream_end_diag",
                          return_value=ok_diag):
            ctx = ps.PostStreamContext(
                proj_sid="sid-E", conv_sid="sid-E", erg_key="sid-E",
                request_id="rid", body={}, cwd="",
                bytes_out=200, status=200, upstream_failed=False,
                keepalives_emitted=0, elapsed=0.2, started=time.time(),
                url="http://x",
            )
            await analyzer.analyze(ctx)
            await asyncio.sleep(0.05)

        info = _erg.peek_force_frontier("sid-E")
        assert info is not None
        assert "empty_response" in info["reason"]
        assert info["completion_tokens"] == 1

    @pytest.mark.asyncio
    async def test_empty_response_writes_forensics_dump(self, tmp_path: Path):
        # Use a nested log_dir so the forensics dir is unique per test
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        cfg = _mk_cfg(log_dir=log_dir)
        buffer = (
            'event: response.completed\n'
            'data: {"usage":{"completion_tokens":0,"finish_reason":"stop"}}\n\n'
        )
        _sc._OUTPUT_BUFFER["sid-F"] = buffer
        analyzer = ps.PostStreamAnalyzer(cfg=cfg, log=lambda *a, **k: None)
        from tinyctx.soft_completion import ClassifyDiag, ClassifyResult
        ok_diag = ClassifyDiag(
            result=ClassifyResult(soft_punt=False, p=0.1, reason="ok"),
        )
        with patch.object(_sc, "classify_at_stream_end_diag",
                          return_value=ok_diag):
            ctx = ps.PostStreamContext(
                proj_sid="sid-F", conv_sid="sid-F", erg_key="sid-F",
                request_id="rid", body={}, cwd="",
                bytes_out=200, status=200, upstream_failed=False,
                keepalives_emitted=0, elapsed=0.2, started=time.time(),
                url="http://x",
            )
            await analyzer.analyze(ctx)
            await asyncio.sleep(0.05)

        forensics_dir = tmp_path / "forensics"
        dumps = list(forensics_dir.glob("*-empty_response-*.json"))
        assert len(dumps) == 1


class TestPostStreamAnalyzerSoftPunt:
    @pytest.mark.asyncio
    async def test_soft_punt_high_p_writes_punt_forensics(self, tmp_path: Path):
        """Classifier returns soft_punt=True with high p -> punt
        forensics dump is written by the BG task."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        cfg = _mk_cfg(log_dir=log_dir)
        buffer = (
            'event: response.completed\n'
            'data: {"usage":{"completion_tokens":50,"finish_reason":"stop"}}\n\n'
        )
        _sc._OUTPUT_BUFFER["sid-P"] = buffer
        analyzer = ps.PostStreamAnalyzer(cfg=cfg, log=lambda *a, **k: None)
        from tinyctx.soft_completion import ClassifyDiag, ClassifyResult
        punt_diag = ClassifyDiag(
            result=ClassifyResult(soft_punt=True, p=0.95,
                                    reason="needs_user_input"),
            extracted_text_chars=120,
            raw_buffer_chars=200,
            finish_reason="stop",
        )
        with patch.object(_sc, "classify_at_stream_end_diag",
                          return_value=punt_diag):
            ctx = ps.PostStreamContext(
                proj_sid="sid-P", conv_sid="sid-P", erg_key="sid-P",
                request_id="rid", body={}, cwd="",
                bytes_out=200, status=200, upstream_failed=False,
                keepalives_emitted=0, elapsed=0.2, started=time.time(),
                url="http://x",
            )
            await analyzer.analyze(ctx)
            # Give the spawned bg classifier task time to run.
            # write_punt_forensics uses trigger="punt_p{int(p*100)}", so
            # match that prefix.
            forensics_dir = tmp_path / "forensics"
            for _ in range(30):
                await asyncio.sleep(0.02)
                if list(forensics_dir.glob("*-punt_p*.json")):
                    break

        dumps = list(forensics_dir.glob("*-punt_p*.json"))
        assert len(dumps) >= 1


# ─── RelayErrorTerminator ─────────────────────────────────────────────────


class TestRelayErrorTerminator:
    def test_stall_cancelled_emits_terminator_and_flags(self, tmp_path: Path):
        from tinyctx import stall_watchdog as _stall
        cfg = _mk_cfg(log_dir=tmp_path)
        streak_counter: dict[str, int] = {"sid": 0}
        term = ps.RelayErrorTerminator(cfg=cfg, log=lambda *a, **k: None,
                                       session_error_streak=streak_counter)
        err = _stall.StallCancelledError("stalled", elapsed_silent_s=120.0,
                                         conv_sid="conv-1")
        result = term.on_stall_cancelled(
            err, proj_sid="sid", conv_sid="conv-1", bytes_out=42,
            started=time.time() - 5.0, url="http://x",
        )
        # Streak incremented
        assert streak_counter["sid"] == 1
        # Result carries: upstream_failed flag, message, status, error_event
        assert result.upstream_failed is True
        assert "stall_cancelled" in result.upstream_failure_msg
        assert result.status == 0
        # Error SSE event contains stall_cancelled type
        assert b"stall_cancelled" in result.error_event

    def test_stall_cancelled_writes_forensics_dump(self, tmp_path: Path):
        from tinyctx import stall_watchdog as _stall
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        cfg = _mk_cfg(log_dir=log_dir)
        term = ps.RelayErrorTerminator(cfg=cfg, log=lambda *a, **k: None,
                                       session_error_streak={})
        err = _stall.StallCancelledError("stalled", elapsed_silent_s=120.0,
                                         conv_sid=None)
        term.on_stall_cancelled(
            err, proj_sid="sid", conv_sid=None, bytes_out=42,
            started=time.time() - 5.0, url="http://x",
        )
        forensics_dir = tmp_path / "forensics"
        dumps = list(forensics_dir.glob("*-stall_cancelled_relay-*.json"))
        assert len(dumps) == 1

    def test_http_error_emits_event_and_writes_forensics(self, tmp_path: Path):
        import httpx
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        cfg = _mk_cfg(log_dir=log_dir)
        streak_counter: dict[str, int] = {"sid": 0}
        term = ps.RelayErrorTerminator(cfg=cfg, log=lambda *a, **k: None,
                                       session_error_streak=streak_counter)
        err = httpx.RemoteProtocolError("conn dropped")
        result = term.on_http_error(
            err, proj_sid="sid", conv_sid=None, bytes_out=10,
            started=time.time() - 1.0, url="http://x",
            erg_key="sid",
        )
        assert streak_counter["sid"] == 1
        assert result.upstream_failed is True
        assert "http error" in result.upstream_failure_msg
        forensics_dir = tmp_path / "forensics"
        dumps = list(forensics_dir.glob("*-stream_error-*.json"))
        assert len(dumps) == 1

    def test_http_error_escalates_after_streak(self, tmp_path: Path):
        """Transient httpx error past upstream_retry_count -> force-frontier
        flag with reason containing 'stream_error escalate'."""
        import httpx
        cfg = _mk_cfg(log_dir=tmp_path)
        # Force escalation immediately by pre-loading streak
        streak_counter: dict[str, int] = {"sid": 3}
        # Cfg needs the retry knobs
        cfg.upstream_retry_enabled = True
        cfg.upstream_retry_count = 1
        term = ps.RelayErrorTerminator(cfg=cfg, log=lambda *a, **k: None,
                                       session_error_streak=streak_counter)
        err = httpx.RemoteProtocolError("conn dropped")
        term.on_http_error(
            err, proj_sid="sid", conv_sid=None, bytes_out=10,
            started=time.time() - 1.0, url="http://x",
            erg_key="sid",
        )
        info = _erg.peek_force_frontier("sid")
        assert info is not None
        assert "stream_error escalate" in info["reason"]


# ─── ForensicsPolicy ──────────────────────────────────────────────────────


class TestForensicsPolicy:
    def test_supported_triggers_list_is_documented(self):
        """The policy class advertises which triggers it handles."""
        triggers = set(ps.ForensicsPolicy.SUPPORTED_TRIGGERS)
        # The actual set the proxy uses today; if any change, this
        # asserts deliberate update.
        assert "punt" in triggers
        assert "punt_via_stream_rewrite" in triggers
        assert "empty_response" in triggers
        assert "stream_error" in triggers
        assert "stall_cancelled_relay" in triggers
        # Any HTTP status code error dump uses "upstream_{code}" — we
        # support the family via a prefix check helper.
        assert ps.ForensicsPolicy.is_supported_trigger("upstream_400")
        assert ps.ForensicsPolicy.is_supported_trigger("upstream_401")
        assert ps.ForensicsPolicy.is_supported_trigger("upstream_500")

    def test_dump_writes_file_with_trigger_in_name(self, tmp_path: Path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        cfg = _mk_cfg(log_dir=log_dir)
        policy = ps.ForensicsPolicy(cfg=cfg, log=lambda *a, **k: None)
        path = policy.dump(
            trigger="empty_response",
            proj_sid="sid",
            response_buffer="some buffer",
            extra={"foo": "bar"},
        )
        assert path is not None
        assert "empty_response" in path.name
        loaded = json.loads(path.read_text())
        assert loaded["trigger"] == "empty_response"
        assert loaded["extra"] == {"foo": "bar"}

    def test_dump_no_op_when_disabled(self, tmp_path: Path):
        cfg = _mk_cfg(log_dir=tmp_path, forensics_enabled=False)
        policy = ps.ForensicsPolicy(cfg=cfg, log=lambda *a, **k: None)
        path = policy.dump(
            trigger="empty_response", proj_sid="sid",
            response_buffer="x",
        )
        assert path is None

    def test_dump_no_op_when_error_capture_disabled_for_errors(self, tmp_path: Path):
        cfg = _mk_cfg(log_dir=tmp_path, forensics_capture_errors=False)
        policy = ps.ForensicsPolicy(cfg=cfg, log=lambda *a, **k: None)
        # Error triggers are gated by forensics_capture_errors
        path = policy.dump(
            trigger="stream_error", proj_sid="sid", response_buffer="x",
        )
        assert path is None

    def test_dump_no_op_when_punt_capture_disabled_for_punts(self, tmp_path: Path):
        cfg = _mk_cfg(log_dir=tmp_path, forensics_capture_punts=False)
        policy = ps.ForensicsPolicy(cfg=cfg, log=lambda *a, **k: None)
        path = policy.dump(
            trigger="punt", proj_sid="sid", response_buffer="x",
        )
        assert path is None
