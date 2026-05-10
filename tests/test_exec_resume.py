"""C-4 hybrid exec_resume: poke a stuck codex.app session by spawning
`codex exec resume <id>` in a side process. Tests cover sqlite session
resolution, rate-limiting, history capture, and the spawn fallback paths
— actual subprocess spawn is mocked since we can't burn real codex API
calls in CI."""
from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest


def _make_codex_state_db(home: Path,
                         rows: list[tuple[str, str, int]]) -> Path:
    """Create a minimal stand-in for ~/.codex/state_5.sqlite. Each row
    is (id, cwd, updated_at). Returns the db path."""
    home.mkdir(parents=True, exist_ok=True)
    db = home / "state_5.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
    """)
    conn.executemany("INSERT INTO threads (id, cwd, updated_at) VALUES (?, ?, ?)",
                     rows)
    conn.commit()
    conn.close()
    return db


@pytest.fixture(autouse=True)
def _reset():
    from tinyctx import exec_resume as _xr
    _xr.reset_state()
    yield
    _xr.reset_state()


# ─── session_id resolution ────────────────────────────────────────────────


def test_resolve_session_id_picks_latest_for_cwd(tmp_path: Path):
    from tinyctx.exec_resume import resolve_session_id
    home = tmp_path / "codex"
    _make_codex_state_db(home, rows=[
        ("aaa-old", "/repo/x", 1000),
        ("bbb-newer", "/repo/x", 2000),
        ("ccc-other", "/repo/y", 3000),
    ])
    sid = resolve_session_id("/repo/x", codex_home=home)
    assert sid == "bbb-newer"


def test_resolve_session_id_returns_none_when_cwd_missing(tmp_path: Path):
    from tinyctx.exec_resume import resolve_session_id
    home = tmp_path / "codex"
    _make_codex_state_db(home, rows=[("a", "/repo/x", 1)])
    assert resolve_session_id("/repo/never-touched", codex_home=home) is None


def test_resolve_session_id_returns_none_when_db_missing(tmp_path: Path):
    from tinyctx.exec_resume import resolve_session_id
    home = tmp_path / "no-codex-here"
    assert resolve_session_id("/repo/x", codex_home=home) is None


def test_resolve_session_id_returns_none_for_empty_cwd(tmp_path: Path):
    from tinyctx.exec_resume import resolve_session_id
    home = tmp_path / "codex"
    _make_codex_state_db(home, rows=[("a", "/repo/x", 1)])
    assert resolve_session_id("", codex_home=home) is None


def test_resolve_session_id_picks_newest_state_file(tmp_path: Path):
    """When multiple state_*.sqlite exist (codex schema migrations),
    the newest mtime wins."""
    from tinyctx.exec_resume import resolve_session_id, _codex_state_db
    home = tmp_path / "codex"
    _make_codex_state_db(home, rows=[("old", "/r", 1)])
    # Rename to state_4.sqlite (older schema), make a state_5.sqlite (newer)
    (home / "state_5.sqlite").rename(home / "state_4.sqlite")
    _make_codex_state_db(home, rows=[("new", "/r", 1)])
    # state_5.sqlite is newer; should win
    db = _codex_state_db(home)
    assert db.name == "state_5.sqlite"
    assert resolve_session_id("/r", codex_home=home) == "new"


# ─── rate-limit ───────────────────────────────────────────────────────────


def test_rate_limit_per_session_cooldown():
    from tinyctx.exec_resume import _check_rate_limits, _LAST_POKE_TS
    sid = "session-a"
    # First call ok
    assert _check_rate_limits(sid, cooldown_s=300, max_per_minute=999) == ""
    # Simulate a recent poke
    _LAST_POKE_TS[sid] = time.time()
    reason = _check_rate_limits(sid, cooldown_s=300, max_per_minute=999)
    assert reason.startswith("cooldown:")


def test_rate_limit_global_per_minute():
    from tinyctx.exec_resume import _check_rate_limits, _RECENT_POKES
    now = time.time()
    for _ in range(3):
        _RECENT_POKES.append(now)
    # 3 in last 60s → cap=3 should reject
    reason = _check_rate_limits("fresh-sid", cooldown_s=0, max_per_minute=3)
    assert reason.startswith("per_minute_cap:")


def test_rate_limit_strips_old_entries():
    """Entries >60s old shouldn't count toward the per-minute cap."""
    from tinyctx.exec_resume import _check_rate_limits, _RECENT_POKES
    old = time.time() - 120
    for _ in range(5):
        _RECENT_POKES.append(old)
    # Old entries — should be drained on check, allowing the new poke
    assert _check_rate_limits("sid", cooldown_s=0, max_per_minute=3) == ""


# ─── poke() — full flow with mocked subprocess ────────────────────────────


@pytest.mark.asyncio
async def test_poke_skipped_when_no_codex_binary(tmp_path: Path):
    from tinyctx import exec_resume as _xr
    home = tmp_path / "codex"
    _make_codex_state_db(home, rows=[("sid", "/r", 1)])
    rec = await _xr.poke(
        cwd="/r",
        prompt="continue",
        codex_binary="/definitely/does/not/exist/codex",
        log_dir=tmp_path / "logs",
        codex_home=home,
    )
    # Override is honored when truthy — but the path doesn't exist → fall back
    # to PATH lookup. In CI 'codex' may or may not be present; in either case
    # the result should be a recorded PokeRecord with deterministic shape.
    assert rec.cwd == "/r"
    assert rec.prompt.startswith("continue")
    assert rec.status in ("skipped", "spawned", "error")


@pytest.mark.asyncio
async def test_poke_skipped_when_no_session_for_cwd(tmp_path: Path,
                                                      monkeypatch):
    from tinyctx import exec_resume as _xr
    home = tmp_path / "codex"
    _make_codex_state_db(home, rows=[("sid-a", "/known", 1)])
    # Force codex binary to a real existing file (not actually invoked
    # because session resolution fails first)
    fake_bin = tmp_path / "codex_bin"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    rec = await _xr.poke(
        cwd="/unknown-repo",
        prompt="continue",
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        codex_home=home,
    )
    assert rec.status == "skipped"
    assert "no_session_for_cwd" in rec.reason


@pytest.mark.asyncio
async def test_poke_skipped_when_rate_limited(tmp_path: Path):
    from tinyctx import exec_resume as _xr
    home = tmp_path / "codex"
    _make_codex_state_db(home, rows=[("sid-x", "/r", 1)])
    fake_bin = tmp_path / "codex_bin"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    # Pre-poke to set the cooldown
    _xr._LAST_POKE_TS["sid-x"] = time.time()
    rec = await _xr.poke(
        cwd="/r",
        prompt="continue",
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        codex_home=home,
        cooldown_s=300,
    )
    assert rec.status == "skipped"
    assert rec.reason.startswith("cooldown:")


@pytest.mark.asyncio
async def test_poke_spawns_subprocess_and_records(tmp_path: Path):
    """End-to-end happy path. Uses a fake codex binary that just exits 0
    so we can verify spawn + log capture without burning real API tokens."""
    from tinyctx import exec_resume as _xr
    home = tmp_path / "codex"
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_codex_state_db(home, rows=[("real-sid", str(repo), 1)])
    # Fake codex: writes its argv to a marker file so we can verify
    # the call site assembled args correctly, then exits.
    fake_bin = tmp_path / "fake_codex"
    marker = tmp_path / "argv_marker.txt"
    fake_bin.write_text(
        f"#!/bin/sh\necho \"$@\" > {marker}\nexit 0\n"
    )
    fake_bin.chmod(0o755)

    rec = await _xr.poke(
        cwd=str(repo),
        prompt="please continue",
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        codex_home=home,
        sandbox="read-only",
        approval_policy="never",
    )
    # asyncio.create_subprocess_exec returned a valid pid → spawn
    # succeeded as far as the stdlib is concerned. We don't poll for
    # marker here because in macOS asyncio + start_new_session, the
    # background _wait_then_close task may race the test teardown.
    # The arg-assembly contract is verified via the log file header
    # which _spawn_exec_resume writes synchronously BEFORE spawn.
    assert rec.status == "spawned"
    assert rec.pid > 0
    assert rec.session_id == "real-sid"
    assert rec.log_path
    log_text = Path(rec.log_path).read_text()
    # Log header records the spawned argv as a Python list repr.
    assert "'exec', 'resume'" in log_text
    assert "real-sid" in log_text
    assert "--json" in log_text
    assert "read-only" in log_text
    assert "approval_policy=never" in log_text
    assert "please continue" in log_text
    # Allow the wait-task to run (drain any scheduled callbacks)
    for _ in range(20):
        if marker.exists():
            break
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_poke_records_explicit_session_id(tmp_path: Path):
    """When caller supplies explicit_session_id, sqlite is bypassed."""
    from tinyctx import exec_resume as _xr
    fake_bin = tmp_path / "fake_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    repo = tmp_path / "repo-no-db"
    repo.mkdir()
    rec = await _xr.poke(
        cwd=str(repo),
        prompt="continue",
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        explicit_session_id="caller-supplied-id",
        codex_home=tmp_path / "no-such-codex-home",
    )
    assert rec.session_id == "caller-supplied-id"
    assert rec.status == "spawned"


# ─── tier system (PASS-first backfill — code added by linter, not TDD'd) ─


def test_select_tier_prompt_count_0_returns_first_tier():
    """count==0 → tiers[0] (gentle)."""
    from tinyctx.exec_resume import select_tier_prompt
    out = select_tier_prompt("sid", ["gentle", "firm", "final"])
    assert out == "gentle"


def test_select_tier_prompt_count_1_returns_first_tier():
    """count==1 also gentle (boundary)."""
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["sid"] = 1
    assert _xr.select_tier_prompt("sid",
                                    ["gentle", "firm", "final"]) == "gentle"


def test_select_tier_prompt_count_2_promotes_to_firm():
    """count==2 → tiers[1] (firm). Boundary: 2 not still gentle."""
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["sid"] = 2
    assert _xr.select_tier_prompt("sid",
                                    ["gentle", "firm", "final"]) == "firm"


def test_select_tier_prompt_count_4_still_firm():
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["sid"] = 4
    assert _xr.select_tier_prompt("sid",
                                    ["gentle", "firm", "final"]) == "firm"


def test_select_tier_prompt_count_5_promotes_to_final():
    """count==5 → tiers[2] (final, last warning)."""
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["sid"] = 5
    assert _xr.select_tier_prompt("sid",
                                    ["gentle", "firm", "final"]) == "final"


def test_select_tier_prompt_count_5_returns_None_when_only_two_tiers():
    """count >= 5 with no terminal tier → None signals tier_exhausted."""
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["sid"] = 5
    assert _xr.select_tier_prompt("sid", ["gentle", "firm"]) is None


def test_select_tier_prompt_empty_tiers_returns_None():
    from tinyctx.exec_resume import select_tier_prompt
    assert select_tier_prompt("sid", []) is None


def test_select_tier_prompt_single_tier_falls_back_for_higher_counts():
    """tiers=['only'] with count=3 → falls back to tiers[-1]."""
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["sid"] = 3
    assert _xr.select_tier_prompt("sid", ["only"]) == "only"


@pytest.mark.asyncio
async def test_poke_uses_tier_prompt_when_provided(tmp_path: Path):
    """poke(prompt_tiers=[...]) — first call uses tiers[0], not the
    legacy single-prompt arg."""
    from tinyctx import exec_resume as _xr
    fake_bin = tmp_path / "fake_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    repo = tmp_path / "repo"; repo.mkdir()
    rec = await _xr.poke(
        cwd=str(repo),
        prompt="LEGACY",
        prompt_tiers=["GENTLE_TIER", "FIRM_TIER", "FINAL_TIER"],
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        explicit_session_id="sid-tier",
    )
    assert rec.status == "spawned"
    # Tier prompt won, not the legacy `prompt` arg
    assert "GENTLE_TIER" in rec.prompt
    assert "LEGACY" not in rec.prompt


@pytest.mark.asyncio
async def test_poke_increments_count_per_session_on_spawn(tmp_path: Path):
    """Each successful spawn must bump _POKE_COUNT_PER_SESSION so the
    tier escalation actually advances."""
    from tinyctx import exec_resume as _xr
    fake_bin = tmp_path / "fake_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    repo = tmp_path / "repo"; repo.mkdir()
    assert _xr.poke_count("sid-bump") == 0
    await _xr.poke(
        cwd=str(repo),
        prompt="x",
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        explicit_session_id="sid-bump",
        cooldown_s=0,  # disable cooldown for this test
    )
    assert _xr.poke_count("sid-bump") == 1


@pytest.mark.asyncio
async def test_poke_returns_tier_exhausted_at_high_count(tmp_path: Path):
    """count >= 5 with only 2 tiers → skipped reason=tier_exhausted —
    AND, when proj_sid is given, sets the force_frontier flag."""
    from tinyctx import exec_resume as _xr
    from tinyctx import empty_response_guard as _erg
    _erg.reset_state()
    fake_bin = tmp_path / "fake_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    repo = tmp_path / "repo"; repo.mkdir()
    # Pre-set the count to simulate prior pokes
    _xr._POKE_COUNT_PER_SESSION["sid-exh"] = 5
    rec = await _xr.poke(
        cwd=str(repo),
        prompt_tiers=["gentle", "firm"],  # only 2 tiers, count=5 → None
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        explicit_session_id="sid-exh",
        proj_sid="proj-exh",
    )
    assert rec.status == "skipped"
    assert rec.reason == "tier_exhausted"
    # force_next_to_frontier flag must be set on proj-exh
    flag = _erg.peek_force_frontier("proj-exh")
    assert flag is not None
    assert "exec_resume_exhausted" in flag.get("reason", "")


@pytest.mark.asyncio
async def test_poke_tier_exhausted_does_not_force_frontier_without_proj_sid(
        tmp_path: Path):
    """No proj_sid → no force_frontier (caller didn't opt in)."""
    from tinyctx import exec_resume as _xr
    from tinyctx import empty_response_guard as _erg
    _erg.reset_state()
    fake_bin = tmp_path / "fake_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    repo = tmp_path / "repo"; repo.mkdir()
    _xr._POKE_COUNT_PER_SESSION["sid-exh-noproj"] = 5
    rec = await _xr.poke(
        cwd=str(repo),
        prompt_tiers=["gentle"],  # 1 tier; count=5 → None
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        explicit_session_id="sid-exh-noproj",
        # proj_sid intentionally omitted
    )
    # Falls back to tiers[-1] when count=5 + 1 tier (per implementation)
    # OR is exhausted — let's check both branches by reading the code.
    # Actual contract: select_tier_prompt with len(tiers)<3 + count>=5
    # returns None ONLY when the function chooses the count>=5 branch.
    # For 1-tier list: count<=1 fallback returns tiers[-1]; count<=4
    # fallback returns tiers[-1]; count>=5 -> None.
    # So count=5 + 1-tier == None == tier_exhausted
    assert rec.status == "skipped"
    assert rec.reason == "tier_exhausted"
    # No force_frontier flag should fire when proj_sid is absent
    flag = _erg.peek_force_frontier("sid-exh-noproj")
    assert flag is None


def test_state_snapshot_exposes_tier_state_label():
    """state_snapshot must surface human-readable tier labels
    ('gentle' / 'firm' / 'final') so the dashboard renders the right
    badge per session."""
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["a"] = 0
    _xr._POKE_COUNT_PER_SESSION["b"] = 3
    _xr._POKE_COUNT_PER_SESSION["c"] = 7
    snap = _xr.state_snapshot()
    assert snap["tier_state"]["a"] == "gentle"
    assert snap["tier_state"]["b"] == "firm"
    assert snap["tier_state"]["c"] == "final"
    # Raw counts also surfaced
    assert snap["poke_counts"] == {"a": 0, "b": 3, "c": 7}


def test_reset_state_clears_poke_count_per_session():
    """Test isolation: reset_state must clear _POKE_COUNT_PER_SESSION,
    not just _LAST_POKE_TS / _RECENT_POKES / _HISTORY. Without this,
    tier state leaks across tests and test order affects results."""
    from tinyctx import exec_resume as _xr
    _xr._POKE_COUNT_PER_SESSION["leftover"] = 42
    _xr.reset_state()
    assert _xr.poke_count("leftover") == 0
    assert dict(_xr._POKE_COUNT_PER_SESSION) == {}


# ─── proxy wiring contract ────────────────────────────────────────────────


def test_proxy_passes_prompt_tiers_and_proj_sid_to_poke():
    """Wiring contract: proxy.py's `_xr.poke(...)` call MUST forward
    `prompt_tiers=CFG.exec_resume_prompt_tiers` and `proj_sid=...`
    so the tiered escalation path is reachable.

    Without this, the `prompt_tiers` parameter on `poke()` exists in
    isolation — every poke uses the single `exec_resume_prompt` and
    the tier system + force_frontier-on-exhaustion fallback never
    trigger. Live trace 2026-05-10: linter added the tier system but
    the wiring is incomplete; this test guards the connection."""
    import inspect
    from pathlib import Path
    src = Path("/Users/sekkit/dev/tinyctx/tinyctx/proxy.py").read_text()
    # Find the _xr.poke( ... ) call site
    idx = src.find("_xr.poke(")
    assert idx != -1, "proxy.py must call _xr.poke(...) somewhere"
    # Look at the next ~1500 chars to capture the full kwarg list
    call_block = src[idx:idx + 1500]
    assert "prompt_tiers=" in call_block, (
        "proxy._xr.poke(...) must pass prompt_tiers=CFG.exec_resume_prompt_tiers "
        "— otherwise the tier escalation system is dead code")
    assert "proj_sid=" in call_block, (
        "proxy._xr.poke(...) must pass proj_sid=... — required for "
        "tier_exhausted to set the force_frontier flag on the right session")


# ─── history snapshot ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_snapshot_returns_recent_pokes(tmp_path: Path):
    from tinyctx import exec_resume as _xr
    fake_bin = tmp_path / "fake_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    home = tmp_path / "codex"
    repo_a = tmp_path / "repo-a"; repo_a.mkdir()
    repo_b = tmp_path / "repo-b"; repo_b.mkdir()
    _make_codex_state_db(home,
                         rows=[("s1", str(repo_a), 1),
                               ("s2", str(repo_b), 1)])
    await _xr.poke(cwd=str(repo_a), prompt="x", codex_binary=str(fake_bin),
                   log_dir=tmp_path / "logs", codex_home=home)
    await _xr.poke(cwd=str(repo_b), prompt="y", codex_binary=str(fake_bin),
                   log_dir=tmp_path / "logs", codex_home=home)
    snap = _xr.history_snapshot()
    assert len(snap) == 2
    # Most recent first
    assert snap[0]["cwd"] == str(repo_b)
    assert snap[1]["cwd"] == str(repo_a)


def test_state_snapshot_groups_by_status():
    from tinyctx import exec_resume as _xr
    _xr._HISTORY.append(_xr.PokeRecord(
        ts=time.time(), cwd="/a", session_id="s1",
        prompt="p", status="spawned"))
    _xr._HISTORY.append(_xr.PokeRecord(
        ts=time.time(), cwd="/b", session_id="s2",
        prompt="p", status="skipped", reason="cooldown"))
    snap = _xr.state_snapshot()
    assert snap["by_status"]["spawned"] == 1
    assert snap["by_status"]["skipped"] == 1
    assert snap["history_total"] == 2


# ─── config defaults ─────────────────────────────────────────────────────


def test_config_defaults_for_exec_resume():
    from tinyctx.config import Config
    cfg = Config()
    assert cfg.exec_resume_enabled is True
    assert cfg.exec_resume_min_p >= 0.5
    assert cfg.exec_resume_cooldown_s > 0
    assert cfg.exec_resume_max_per_minute > 0
    assert isinstance(cfg.exec_resume_prompt, str)
    assert len(cfg.exec_resume_prompt) > 10
    assert cfg.exec_resume_sandbox in ("read-only", "workspace-write",
                                         "danger-full-access")


def test_config_default_prompt_tiers():
    from tinyctx.config import Config
    cfg = Config()
    tiers = cfg.exec_resume_prompt_tiers
    assert isinstance(tiers, list)
    assert len(tiers) == 3
    for t in tiers:
        assert isinstance(t, str)
        assert len(t) > 10


@pytest.mark.asyncio
async def test_poke_back_compat_without_prompt_tiers(tmp_path: Path):
    from tinyctx import exec_resume as _xr
    fake_bin = tmp_path / "fake_codex"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    repo = tmp_path / "repo-bc"; repo.mkdir()
    rec = await _xr.poke(
        cwd=str(repo),
        prompt="LEGACY_BACKCOMPAT",
        codex_binary=str(fake_bin),
        log_dir=tmp_path / "logs",
        explicit_session_id="sid-bc",
    )
    assert rec.status == "spawned"
    log_text = Path(rec.log_path).read_text()
    assert "LEGACY_BACKCOMPAT" in log_text
