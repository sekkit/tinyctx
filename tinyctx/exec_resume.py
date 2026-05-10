"""C-4 hybrid: poke a stuck codex.app session by spawning `codex exec
resume <session_id> "<prompt>"` in a side process.

Why
───
codex.app pauses on `finish_reason=stop` with no tool_call — it's the
"agent finished, waiting for user input" state. tinyctx detects this
(empty_response_guard / soft_completion classifier) and sets the
`force_next_to_frontier` flag. BUT the flag only fires on the NEXT
request from codex.app, and codex.app won't make one until the user
types. The user (especially while away from keyboard) never types →
session stalls indefinitely. That's the recurring "又断了" problem.

C-4's elegance is that `codex exec` is a ONE-SHOT non-interactive mode:
no "finish_reason=stop wait for user" concept. `codex exec resume <id>
"<prompt>"` loads the same session_id from disk, appends the prompt as
a new user message, and executes one full turn end-to-end. The new
turn writes back to the SAME jsonl rollout + sqlite (codex.app's
storage), so codex.app's UI eventually surfaces it.

This module fires that side process when the soft_completion classifier
returns a high-confidence PUNT — turning a passive flag into an
active nudge that doesn't wait on the user.

Wiring
──────
proxy.py (_bg_classify high-confidence PUNT branch) →
  exec_resume.poke(cwd=request_cwd, p=verdict_p)

poke():
  1. Resolve session_id from cwd via ~/.codex/state_5.sqlite
     (latest thread.id for that cwd)
  2. Check rate-limit (per-session cooldown + global per-minute)
  3. Spawn `codex exec resume <id> "<CFG.exec_resume_prompt>"`
     in a fully-detached subprocess; capture stdout/stderr to a logfile
  4. Append to in-memory poke history (for dashboard)
  5. Return immediately — don't wait for the subprocess

Concurrency safety
──────────────────
codex.app holds the sqlite file (WAL mode — visible from `state_5.sqlite-wal`
in the codex home), and `codex exec resume` opens the same DB. WAL allows
multiple readers + one writer concurrently, so the side process won't
deadlock the main app. The new turn is written through the standard
codex code paths and shows up in `~/.codex/sessions/.../*.jsonl`
(codex.app's source of truth).

Approval / sandbox
──────────────────
The poked turn runs with `-s read-only` + `approval_policy=never`
by default — agent can think + plan + call advisor + run trusted-set
read-only commands, but cannot modify files. This is the conservative
"give it a nudge to continue thinking, but don't let it loose" setting.
codex.app users can flip `exec_resume_sandbox` in config.toml to
`workspace-write` once they trust the loop.

Failure modes (silent — never raises into the proxy)
────────────────────────────────────────────────────
- codex binary not on PATH → log + skip
- session_id can't be resolved → log + skip
- rate-limit exceeded → log + skip
- subprocess spawn fails → log + skip

Each failure is recorded in the poke history so the dashboard surfaces
it; nothing crashes the proxy.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── default codex paths ───────────────────────────────────────────────────


# codex.app on macOS ships at /Applications/Codex.app/Contents/Resources/codex.
# CLI install via `codex` on PATH also works. Tried in order.
_CODEX_BIN_CANDIDATES = (
    "/Applications/Codex.app/Contents/Resources/codex",
    "/usr/local/bin/codex",
    "/opt/homebrew/bin/codex",
)


def _resolve_codex_binary(override: str = "") -> str:
    """Pick the codex binary path. Override wins; else first candidate
    that exists on disk; else falls back to `which codex`. Returns
    empty string when nothing is found (caller logs + skips)."""
    if override:
        return override
    for c in _CODEX_BIN_CANDIDATES:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    found = shutil.which("codex")
    return found or ""


# ─── session_id resolution from sqlite ────────────────────────────────────


def _codex_state_db(codex_home: Path) -> Path:
    """The sqlite file codex.app writes its thread metadata to. Lives
    under $CODEX_HOME (default ~/.codex). The numeric suffix is a
    schema-migration revision; codex auto-bumps it when migrations
    change shape — most recent file wins."""
    matches = sorted(codex_home.glob("state_*.sqlite"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    # Fallback to the canonical name even if it doesn't exist yet —
    # caller's `if not path.exists()` will skip cleanly.
    return codex_home / "state_5.sqlite"


def resolve_session_id(cwd: str,
                        codex_home: Path | None = None) -> str | None:
    """Return the most recent thread.id whose cwd matches `cwd`. None
    when no match (or sqlite unreadable). Read-only — opens the DB
    in URI mode with `mode=ro` so we never risk corrupting codex's
    write path."""
    if not cwd:
        return None
    home = codex_home or (Path.home() / ".codex")
    db = _codex_state_db(home)
    if not db.exists():
        return None
    try:
        # URI form keeps the read strictly read-only. `nolock=1` was
        # tempting (avoid blocking on codex.app's WAL writer) but
        # macOS sqlite refuses to open the DB with that combination —
        # `mode=ro` alone is sufficient since WAL allows multi-reader
        # concurrent with one writer, and verified empirically against
        # codex.app's live state_5.sqlite (journal_mode=wal).
        conn = sqlite3.connect(
            f"file:{db}?mode=ro",
            uri=True, timeout=2.0)
    except sqlite3.Error:
        return None
    try:
        cur = conn.execute(
            "SELECT id FROM threads WHERE cwd = ? "
            "ORDER BY updated_at DESC LIMIT 1",
            (cwd,))
        row = cur.fetchone()
    except sqlite3.Error:
        return None
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return row[0] if row else None


# ─── rate-limit + history ─────────────────────────────────────────────────


@dataclass
class PokeRecord:
    """One poke attempt — outcome + metadata for dashboard display."""
    ts: float
    cwd: str
    session_id: str
    prompt: str
    status: str            # "spawned" / "skipped" / "error"
    reason: str = ""       # human-readable ("rate_limited" / etc.)
    pid: int = 0
    log_path: str = ""     # subprocess stdout/stderr capture file


# Per-session cooldown timestamps + global per-minute counter.
_LAST_POKE_TS: dict[str, float] = defaultdict(float)  # by session_id
_RECENT_POKES: deque[float] = deque(maxlen=120)       # global timestamps
_HISTORY: deque[PokeRecord] = deque(maxlen=50)        # for dashboard
# Per-session count of successful pokes — drives the tiered prompt
# escalation (gentle → firm → final → tier_exhausted).
_POKE_COUNT_PER_SESSION: dict[str, int] = defaultdict(int)


def poke_count(session_id: str) -> int:
    return _POKE_COUNT_PER_SESSION.get(session_id, 0)


def select_tier_prompt(session_id: str,
                        tiers: list[str]) -> str | None:
    """Pick which tier prompt to send based on prior poke count for
    this session.

    Tier table:
      count 0..1 -> tiers[0]   (fall back to tiers[-1] if missing)
      count 2..4 -> tiers[1]   (fall back to tiers[-1] if missing)
      count >= 5 -> tiers[2]   (None if len(tiers) < 3 — caller stops)
    """
    if not tiers:
        return None
    count = _POKE_COUNT_PER_SESSION.get(session_id, 0)
    if count <= 1:
        return tiers[0] if len(tiers) >= 1 else tiers[-1]
    if count <= 4:
        return tiers[1] if len(tiers) >= 2 else tiers[-1]
    if len(tiers) >= 3:
        return tiers[2]
    return None


def _check_rate_limits(session_id: str,
                        cooldown_s: int,
                        max_per_minute: int) -> str:
    """Return empty string if poke is allowed, else a reason string
    (used as PokeRecord.reason on rejection)."""
    now = time.time()
    last = _LAST_POKE_TS.get(session_id, 0.0)
    if last and (now - last) < cooldown_s:
        return f"cooldown: {int(cooldown_s - (now - last))}s remaining"
    # Global per-minute cap — strip stale entries first
    cutoff = now - 60.0
    while _RECENT_POKES and _RECENT_POKES[0] < cutoff:
        _RECENT_POKES.popleft()
    if len(_RECENT_POKES) >= max_per_minute:
        return f"per_minute_cap: {max_per_minute}/min reached"
    return ""


# ─── subprocess spawn (async, fully detached) ─────────────────────────────


async def _spawn_exec_resume(
        codex_bin: str,
        session_id: str,
        cwd: str,
        prompt: str,
        log_dir: Path,
        sandbox: str,
        approval_policy: str,
        timeout_s: int,
) -> tuple[int, str]:
    """Start `codex exec resume <id>` and capture stdout/stderr to a
    log file. Returns (pid, log_path). Subprocess runs to completion in
    the background — this coroutine awaits its exit but does NOT block
    the caller (it's spawned via asyncio.create_task). Timeout kills
    the subprocess; partial log is preserved.

    Working directory is set to `cwd` so any tool calls the agent makes
    target the right repo."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts_token = time.strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{ts_token}-{session_id[:8]}.log"

    args = [
        codex_bin, "exec", "resume", session_id,
        "--json",
        "--skip-git-repo-check",
        "-s", sandbox,
        "-c", f"approval_policy={approval_policy}",
        prompt,
    ]
    # Open log file BEFORE spawn so subprocess writes hit disk immediately.
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    log_fh.write(
        f"# tinyctx exec_resume poke\n"
        f"# ts={ts_token} session_id={session_id} cwd={cwd}\n"
        f"# args={args!r}\n# ---\n")
    log_fh.flush()

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd or None,
            stdout=log_fh,
            stderr=log_fh,
            stdin=asyncio.subprocess.DEVNULL,
            # Detach from parent's controlling terminal — codex exec
            # otherwise tries to use TTY for status updates and fails
            # ("not a tty") in our background context.
            start_new_session=True,
        )
    except (FileNotFoundError, PermissionError, OSError) as e:
        log_fh.write(f"# spawn failed: {e}\n")
        log_fh.close()
        return 0, str(log_path)

    pid = proc.pid

    async def _wait_then_close():
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            log_fh.write(f"\n# tinyctx: timed out after {timeout_s}s, killed\n")
        except Exception as e:  # noqa: BLE001
            log_fh.write(f"\n# tinyctx: wait error: {e}\n")
        finally:
            try:
                log_fh.write(f"\n# exit_code={proc.returncode}\n")
                log_fh.close()
            except Exception:  # noqa: BLE001
                pass

    asyncio.create_task(_wait_then_close())
    return pid, str(log_path)


# ─── public entry point ───────────────────────────────────────────────────


async def poke(
        cwd: str,
        *,
        prompt: str = "",
        prompt_tiers: list[str] | None = None,
        codex_binary: str = "",
        sandbox: str = "read-only",
        approval_policy: str = "never",
        cooldown_s: int = 300,
        max_per_minute: int = 3,
        timeout_s: int = 60,
        log_dir: Path | None = None,
        codex_home: Path | None = None,
        explicit_session_id: str = "",
        proj_sid: str = "",
) -> PokeRecord:
    """Fire a `codex exec resume` side process for the codex.app
    session whose cwd matches `cwd`. Returns the PokeRecord regardless
    of outcome (status="spawned" / "skipped" / "error").

    Never raises — proxy callers can ignore the return value. The
    in-memory history (`_HISTORY`) records every attempt for the
    dashboard.

    `explicit_session_id` overrides the sqlite lookup — useful for
    tests or when the caller already knows the id from a request
    header. Otherwise falls back to the latest-thread-for-cwd query.

    `prompt_tiers` activates the SPEC §12.3-style tiered prompt
    escalation: `select_tier_prompt(session_id, prompt_tiers)` resolves
    the prompt to send, falling back through tiers as count grows. When
    that helper returns `None` (count >= 5 with no terminal tier) the
    poke is skipped with reason `tier_exhausted` and — when `proj_sid`
    is supplied — `empty_response_guard.force_next_to_frontier` fires so
    the next request from codex routes to frontier. Pass `prompt` for
    back-compat single-prompt callers.
    """
    rec = PokeRecord(
        ts=time.time(), cwd=cwd, session_id="",
        prompt=prompt[:200], status="error", reason="")

    # Step 1: codex binary
    bin_path = _resolve_codex_binary(codex_binary)
    if not bin_path:
        rec.status = "skipped"
        rec.reason = "codex_binary_not_found"
        _HISTORY.append(rec)
        return rec

    # Step 2: session_id
    session_id = explicit_session_id or (
        resolve_session_id(cwd, codex_home=codex_home) or "")
    if not session_id:
        rec.status = "skipped"
        rec.reason = f"no_session_for_cwd: {cwd[:60]}"
        _HISTORY.append(rec)
        return rec
    rec.session_id = session_id

    # Step 3: tier resolution (when caller opted in). Done BEFORE
    # rate-limits so a tier-exhausted session is recorded with that
    # specific reason rather than masked by a stale cooldown.
    effective_prompt = prompt
    if prompt_tiers is not None:
        chosen = select_tier_prompt(session_id, prompt_tiers)
        if chosen is None:
            rec.status = "skipped"
            rec.reason = "tier_exhausted"
            if proj_sid:
                try:
                    from . import empty_response_guard as _erg
                    _erg.force_next_to_frontier(
                        proj_sid, "exec_resume_exhausted")
                except Exception:  # noqa: BLE001
                    pass
            _HISTORY.append(rec)
            return rec
        effective_prompt = chosen
        rec.prompt = effective_prompt[:200]

    # Step 4: rate-limits
    rl_reason = _check_rate_limits(session_id, cooldown_s, max_per_minute)
    if rl_reason:
        rec.status = "skipped"
        rec.reason = rl_reason
        _HISTORY.append(rec)
        return rec

    # Step 5: spawn (default log dir mirrors forensics location)
    log_root = log_dir or (Path.home() / ".tinyctx" / "exec_resume_logs")
    try:
        pid, log_path = await _spawn_exec_resume(
            bin_path, session_id, cwd, effective_prompt, log_root,
            sandbox, approval_policy, timeout_s)
    except Exception as e:  # noqa: BLE001
        rec.reason = f"spawn_exception: {type(e).__name__}: {e!s}"[:200]
        _HISTORY.append(rec)
        return rec

    if pid == 0:
        rec.reason = "spawn_returned_pid_0"
        rec.log_path = log_path
        _HISTORY.append(rec)
        return rec

    # Success — record + update rate-limit state
    rec.status = "spawned"
    rec.pid = pid
    rec.log_path = log_path
    _LAST_POKE_TS[session_id] = rec.ts
    _RECENT_POKES.append(rec.ts)
    _POKE_COUNT_PER_SESSION[session_id] += 1
    _HISTORY.append(rec)
    return rec


# ─── dashboard helpers ────────────────────────────────────────────────────


def history_snapshot(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent poke attempts as plain dicts. Dashboard renders this."""
    out: list[dict[str, Any]] = []
    for r in list(_HISTORY)[-limit:]:
        out.append({
            "ts": round(r.ts, 1),
            "cwd": r.cwd,
            "session_id": r.session_id,
            "prompt": r.prompt,
            "status": r.status,
            "reason": r.reason,
            "pid": r.pid,
            "log_path": r.log_path,
        })
    return list(reversed(out))  # most recent first


def state_snapshot() -> dict[str, Any]:
    """Compact summary: counts by status + active cooldowns."""
    by_status: dict[str, int] = defaultdict(int)
    for r in _HISTORY:
        by_status[r.status] += 1
    now = time.time()
    cooldowns = {sid: round(now - ts, 1)
                 for sid, ts in _LAST_POKE_TS.items()
                 if (now - ts) < 600}  # show last 10 minutes
    poke_counts = dict(_POKE_COUNT_PER_SESSION)
    tier_state = {
        sid: ("gentle" if c <= 1
              else "firm" if c <= 4
              else "final")
        for sid, c in poke_counts.items()
    }
    return {
        "history_total": len(_HISTORY),
        "by_status": dict(by_status),
        "recent_pokes_per_min": len([t for t in _RECENT_POKES
                                       if now - t < 60]),
        "cooldowns_age_s": cooldowns,
        "poke_counts": poke_counts,
        "tier_state": tier_state,
    }


# ─── test/dev helpers ─────────────────────────────────────────────────────


def reset_state() -> None:
    """Clear all module state. Test helper."""
    _LAST_POKE_TS.clear()
    _RECENT_POKES.clear()
    _HISTORY.clear()
    _POKE_COUNT_PER_SESSION.clear()
