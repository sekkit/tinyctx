"""Graphify auto-install + auto-wire bootstrap.

Graphify (safishamsi/graphify, PyPI `graphifyy`, MIT) is a tree-sitter
code knowledge-graph + skill that codex invokes via PreToolUse hook +
AGENTS.md instructions. tinyctx auto-enables it so codex sessions get
the graph map without manual setup, mirroring the gitnexus integration.

Unlike gitnexus, graphify is NOT a stdio MCP server — it's a codex skill.
The integration happens in two layers:

  1. Global  — install the `graphify` binary on PATH (once per machine).
  2. Project — `graphify codex install` writes a `## graphify` section to
     the project's AGENTS.md and registers a PreToolUse hook in the
     project's `.codex/hooks.json`. Both writes are fully idempotent
     (verified: re-runs print "already configured").

Install path priority (only first available is used):
  a. Already on PATH      → skip
  b. uv tool install      → preferred (Astral, manages own Python)
  c. pipx install         → second choice
  d. python3.10+ -m pip   → if a system python ≥3.10 exists
  e. uv-via-curl (opt-in) → bootstraps `uv` itself when nothing else works

Failure handling: every step logs to ~/.tinyctx/logs/graphify-bootstrap.log
and never raises. Bootstrap may finish with `success=False` but proxy
startup is unaffected.

CLI:
    python -m tinyctx.graphify_bootstrap status
    python -m tinyctx.graphify_bootstrap install
    python -m tinyctx.graphify_bootstrap install --project /path/to/repo
    python -m tinyctx.graphify_bootstrap install --all-registered
    python -m tinyctx.graphify_bootstrap uninstall

Env vars:
    TINYCTX_GRAPHIFY_DISABLE=1        bypass everything
    TINYCTX_GRAPHIFY_PKG=graphifyy    pin a different PyPI distribution name
    TINYCTX_GRAPHIFY_AUTO_UV=1        allow uv-via-curl bootstrap (default 1)
    TINYCTX_GRAPHIFY_BIN=...          force a specific binary path
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


GRAPHIFY_BIN = "graphify"
GRAPHIFY_PYPI_PKG = os.environ.get("TINYCTX_GRAPHIFY_PKG", "graphifyy")
GRAPHIFY_GIT_URL = "https://github.com/safishamsi/graphify.git"
# Optional override: when set, install from `git+<url>@<ref>` instead of PyPI.
# Useful when the user wants the latest main HEAD or a specific commit/branch.
# Empty = use PyPI (default; PyPI auto-publishes within ~1 min of git tags).
# IMPORTANT: do NOT default to "v1.0.0" — that tag predates many fixes and
# its pyproject.toml still says version=0.1.10. The author's tag naming is
# inconsistent; rely on PyPI for the canonical "latest stable".
GRAPHIFY_GIT_REF = os.environ.get("TINYCTX_GRAPHIFY_GIT_REF", "")
MIN_PYTHON_MAJOR = 3
MIN_PYTHON_MINOR = 10

UV_INSTALL_URL = "https://astral.sh/uv/install.sh"

TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "graphify-bootstrap.log"
PROJECT_INSTALL_MARKER = ".codex/hooks.json"  # graphify writes this
AGENTS_MARKER = "## graphify"


# ───────────────────────── logging ─────────────────────────


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        # Why: _log itself must never raise — bootstrap log is advisory,
        # not load-bearing for behavior.
        pass


def _which(cmd: str) -> str:
    from .mcp_registry import _which_with_fallbacks as _wf
    return _wf(cmd) or ""


# ───────────────────────── detection ─────────────────────────


@dataclass
class State:
    disabled: bool = False
    graphify_path: str = ""
    graphify_present: bool = False
    uv_path: str = ""
    pipx_path: str = ""
    python_310_plus_path: str = ""
    auto_uv_allowed: bool = True


def _python_meets_min(p: Path) -> bool:
    """Probe a python interpreter and check it's ≥3.10."""
    if not p.is_file() or not os.access(p, os.X_OK):
        return False
    try:
        r = subprocess.run([str(p), "-c",
                            "import sys; print(sys.version_info[:2])"],
                           capture_output=True, text=True, timeout=5)
        m = re.search(r"\((\d+),\s*(\d+)\)", r.stdout)
        if not m:
            return False
        major, minor = int(m.group(1)), int(m.group(2))
        return (major, minor) >= (MIN_PYTHON_MAJOR, MIN_PYTHON_MINOR)
    except (subprocess.SubprocessError, OSError):
        return False


def _find_python_310_plus() -> str:
    """Search common locations for python ≥3.10. Returns abs path or ''."""
    # Try named binaries first
    for ver in ("3.13", "3.12", "3.11", "3.10"):
        for prefix in ("python", "python3."):
            cand = _which(f"{prefix}{ver}" if prefix.endswith(".") else
                          f"{prefix}{ver}")
            if cand and _python_meets_min(Path(cand)):
                return cand
        # also without dot: python3.10 vs python3.10 (same)
        cand = _which(f"python{ver}")
        if cand and _python_meets_min(Path(cand)):
            return cand
    # Try plain python3 last
    cand = _which("python3")
    if cand and _python_meets_min(Path(cand)):
        return cand
    return ""


def detect_state() -> State:
    s = State()
    s.disabled = os.environ.get("TINYCTX_GRAPHIFY_DISABLE", "0") == "1"
    s.auto_uv_allowed = os.environ.get("TINYCTX_GRAPHIFY_AUTO_UV", "1") != "0"

    forced_bin = os.environ.get("TINYCTX_GRAPHIFY_BIN")
    if forced_bin and Path(forced_bin).is_file():
        s.graphify_path = forced_bin
    else:
        s.graphify_path = _which(GRAPHIFY_BIN)
    s.graphify_present = bool(s.graphify_path)

    s.uv_path = _which("uv")
    s.pipx_path = _which("pipx")
    s.python_310_plus_path = _find_python_310_plus()
    return s


# ───────────────────────── installers ────────────────────────


def _run(cmd: list[str], *, timeout: int = 600) -> tuple[bool, str]:
    """Run a command, capture output, return (ok, summary)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except OSError as e:
        return False, f"OSError: {e}"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-400:].strip()
        return False, f"rc={r.returncode}: {tail}"
    return True, "ok"


def _resolve_install_source() -> tuple[list[str], str]:
    """Return (uv-tool-args, label).

    When TINYCTX_GRAPHIFY_GIT_REF is set, install from git@ref:
        uv tool install --from git+<url>@<ref> graphifyy
    Otherwise install from PyPI:
        uv tool install graphifyy
    """
    if GRAPHIFY_GIT_REF:
        return ([
            "--from",
            f"git+{GRAPHIFY_GIT_URL}@{GRAPHIFY_GIT_REF}",
            GRAPHIFY_PYPI_PKG,
        ], f"git@{GRAPHIFY_GIT_REF}")
    return ([GRAPHIFY_PYPI_PKG], "pypi")


def install_via_uv(state: State, *, dry_run: bool = False
                   ) -> tuple[bool, str]:
    if not state.uv_path:
        return False, "uv not on PATH"
    args, label = _resolve_install_source()
    cmd = [state.uv_path, "tool", "install", *args]
    if dry_run:
        return True, f"DRY-RUN ({label}): " + " ".join(cmd)
    ok, msg = _run(cmd, timeout=600)
    return ok, f"{label}: {msg}"


def _pip_target_arg() -> str:
    """Return the pip-style requirement spec.
    PyPI: "graphifyy"
    Git:  "graphifyy @ git+https://github.com/safishamsi/graphify.git@<ref>"
    """
    if GRAPHIFY_GIT_REF:
        return (f"{GRAPHIFY_PYPI_PKG} @ git+{GRAPHIFY_GIT_URL}"
                f"@{GRAPHIFY_GIT_REF}")
    return GRAPHIFY_PYPI_PKG


def install_via_pipx(state: State, *, dry_run: bool = False
                     ) -> tuple[bool, str]:
    if not state.pipx_path:
        return False, "pipx not on PATH"
    target = _pip_target_arg()
    cmd = [state.pipx_path, "install", target]
    if dry_run:
        return True, "DRY-RUN: " + " ".join(cmd)
    return _run(cmd, timeout=600)


def install_via_pip_user(state: State, *, dry_run: bool = False
                         ) -> tuple[bool, str]:
    if not state.python_310_plus_path:
        return False, "no python ≥3.10 on PATH"
    target = _pip_target_arg()
    cmd = [state.python_310_plus_path, "-m", "pip", "install",
           "--user", target]
    if dry_run:
        return True, "DRY-RUN: " + " ".join(cmd)
    return _run(cmd, timeout=600)


def bootstrap_uv_via_curl(*, dry_run: bool = False
                          ) -> tuple[bool, str]:
    """Bootstrap Astral's uv via the official curl-pipe installer.
    Only invoked when no other installer is available AND user has
    opted in (TINYCTX_GRAPHIFY_AUTO_UV=1, default)."""
    if dry_run:
        return True, f"DRY-RUN: curl -LsSf {UV_INSTALL_URL} | sh"
    if not _which("curl"):
        return False, "curl missing; cannot bootstrap uv"
    if not _which("sh"):
        return False, "sh missing; cannot bootstrap uv"
    # Two-step (download then exec) so we don't blindly pipe to sh.
    tmp_script = Path(f"/tmp/tinyctx-uv-install-{os.getpid()}.sh")
    try:
        ok, msg = _run(["curl", "-LsSf", UV_INSTALL_URL, "-o", str(tmp_script)],
                       timeout=120)
        if not ok:
            return False, f"download failed: {msg}"
        ok, msg = _run(["sh", str(tmp_script)], timeout=180)
        if not ok:
            return False, f"installer failed: {msg}"
        # Clean up
        try:
            tmp_script.unlink()
        except OSError as e:
            # Why: best-effort temp cleanup after successful install.
            _log(f"tmp installer unlink failed: {type(e).__name__}: {e}")
        return True, "uv bootstrapped to ~/.local/bin/uv"
    finally:
        if tmp_script.is_file():
            try:
                tmp_script.unlink()
            except OSError as e:
                # Why: best-effort finally-block cleanup; the install
                # outcome is already decided by this point.
                _log(f"tmp installer unlink (finally) failed: {type(e).__name__}: {e}")


def install_globally(state: State, *, dry_run: bool = False
                     ) -> tuple[bool, str, str]:
    """Try the installer chain. Returns (ok, summary, installer_used).

    Priority:
      1. uv tool install        (cleanest, uv manages Python)
      2. pipx install
      3. python3.10+ pip --user
      4. bootstrap uv via curl, then uv tool install (gated by auto_uv_allowed)
    """
    # 1. uv
    if state.uv_path:
        ok, msg = install_via_uv(state, dry_run=dry_run)
        if ok:
            return True, msg, "uv"
        _log(f"uv install failed: {msg}")
    # 2. pipx
    if state.pipx_path:
        ok, msg = install_via_pipx(state, dry_run=dry_run)
        if ok:
            return True, msg, "pipx"
        _log(f"pipx install failed: {msg}")
    # 3. pip --user
    if state.python_310_plus_path:
        ok, msg = install_via_pip_user(state, dry_run=dry_run)
        if ok:
            return True, msg, "pip-user"
        _log(f"pip-user install failed: {msg}")
    # 4. bootstrap uv via curl
    if state.auto_uv_allowed and not state.uv_path:
        ok, msg = bootstrap_uv_via_curl(dry_run=dry_run)
        if not ok:
            return False, f"all installers failed (last: {msg})", ""
        # re-detect uv now and try install
        new_state = detect_state()
        if not new_state.uv_path:
            return False, "uv bootstrap reported success but uv still not found", ""
        ok, msg = install_via_uv(new_state, dry_run=dry_run)
        return ok, f"uv bootstrap + install: {msg}", "uv-bootstrapped"
    return False, "no installer available", ""


# ───────────────────────── per-project install ───────────────


def install_into_project(project_root: Path, state: State, *,
                         dry_run: bool = False) -> tuple[bool, str]:
    """Run `graphify codex install` in `project_root`. Idempotent.

    Returns (ok, message). Failure reasons logged.
    """
    if not state.graphify_path:
        return False, "graphify binary not available"
    if not project_root.is_dir():
        return False, f"project root missing: {project_root}"

    if dry_run:
        return True, (f"DRY-RUN would run: {state.graphify_path} "
                      f"codex install (cwd={project_root})")

    cmd = [state.graphify_path, "codex", "install"]
    try:
        r = subprocess.run(cmd, cwd=str(project_root),
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "graphify codex install timed out"
    except OSError as e:
        return False, f"OSError: {e}"
    out = (r.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    if r.returncode != 0:
        return False, f"rc={r.returncode}: {(r.stderr or '')[:300]}"
    # graphify prints either "graphify section written to ..."  (first run)
    # or                     "graphify already configured in AGENTS.md" (re-run)
    if "already configured" in (r.stdout or ""):
        return True, "already installed"
    return True, last or "ok"


# ───────────────────────── orchestration ─────────────────────


@dataclass
class BootstrapReport:
    state_before: dict = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    success: bool = False


def bootstrap(*, install: bool = True,
              project_roots: list[Path] | None = None,
              dry_run: bool = False) -> BootstrapReport:
    actions: list[str] = []
    skipped: list[str] = []

    state = detect_state()
    state_before = asdict(state)
    _log(f"start state={state_before} dry_run={dry_run} "
         f"projects={project_roots}")

    if state.disabled:
        skipped.append("TINYCTX_GRAPHIFY_DISABLE=1")
        _log("disabled by env; abort")
        final = detect_state()
        return BootstrapReport(state_before, actions, asdict(final),
                               skipped, success=True)

    if install and not state.graphify_present:
        ok, msg, installer = install_globally(state, dry_run=dry_run)
        actions.append(f"global install via {installer or 'none'}: {msg}")
        _log(f"install: {ok} via {installer}: {msg}")
        if ok and not dry_run:
            state = detect_state()
    elif state.graphify_present:
        actions.append(f"global install: already at {state.graphify_path}")

    if project_roots and state.graphify_present:
        for root in project_roots:
            ok, msg = install_into_project(root, state, dry_run=dry_run)
            mark = "✓" if ok else "✗"
            actions.append(f"project {root.name}: {mark} {msg}")
            _log(f"project {root}: {ok}: {msg}")
    elif project_roots and not state.graphify_present:
        skipped.append("project install skipped: graphify binary missing")

    final = detect_state()
    success = all(("✗" not in a) for a in actions)
    _log(f"final state={asdict(final)} success={success}")
    return BootstrapReport(state_before, actions, asdict(final),
                           skipped, success)


# ───────────────────────────── CLI ───────────────────────────


def _print_state_human(state: State) -> None:
    rows = [
        ("disabled (env)", "yes" if state.disabled else "no"),
        ("graphify binary", state.graphify_path or "missing"),
        ("uv", state.uv_path or "missing"),
        ("pipx", state.pipx_path or "missing"),
        ("python ≥3.10", state.python_310_plus_path or "missing"),
        ("auto-uv allowed", "yes" if state.auto_uv_allowed else "no"),
    ]
    print("graphify state:")
    for k, v in rows:
        print(f"  {k:<22}  {v}")


def _registered_projects() -> list[Path]:
    try:
        from . import registry
        return list(registry.all_projects())
    except Exception:  # noqa: BLE001
        return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tinyctx.graphify_bootstrap")
    p.add_argument("cmd", nargs="?", default="install",
                   choices=["install", "status", "uninstall", "global-only"])
    p.add_argument("--project", action="append", default=[],
                   help="project root to wire (repeatable). Defaults to cwd "
                        "for `install` if neither --project nor "
                        "--all-registered is given.")
    p.add_argument("--all-registered", action="store_true",
                   help="iterate every project in the tinyctx registry")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "status":
        _print_state_human(detect_state())
        return 0

    if args.cmd == "uninstall":
        return _cmd_uninstall(args)

    project_roots: list[Path] = [Path(p).resolve() for p in args.project]
    if args.all_registered:
        project_roots.extend(_registered_projects())
    if args.cmd == "install" and not project_roots:
        project_roots = [Path.cwd().resolve()]
    if args.cmd == "global-only":
        project_roots = []  # explicit: global only

    do_install = args.cmd in ("install", "global-only")
    report = bootstrap(install=do_install,
                       project_roots=project_roots,
                       dry_run=args.dry_run)

    if not args.quiet:
        print(f"[graphify-bootstrap] success={report.success}")
        for a in report.actions:
            print(f"  {a}")
        for s in report.skipped:
            print(f"  ⏭ {s}", file=sys.stderr)
    return 0 if report.success else 1


def _cmd_uninstall(args) -> int:
    """Per-project uninstall via `graphify codex uninstall`. Does NOT
    remove the global binary (user may want it independently)."""
    state = detect_state()
    if not state.graphify_present:
        print("graphify not installed globally; nothing to uninstall",
              file=sys.stderr)
        return 0
    project_roots: list[Path] = [Path(p).resolve() for p in args.project]
    if args.all_registered:
        project_roots.extend(_registered_projects())
    if not project_roots:
        project_roots = [Path.cwd().resolve()]
    failed = 0
    for root in project_roots:
        try:
            r = subprocess.run([state.graphify_path, "codex", "uninstall"],
                               cwd=str(root), capture_output=True,
                               text=True, timeout=30)
            if r.returncode != 0:
                print(f"  ✗ {root.name}: rc={r.returncode}", file=sys.stderr)
                failed += 1
            else:
                print(f"  ✓ {root.name}: uninstalled")
        except (subprocess.SubprocessError, OSError) as e:
            print(f"  ✗ {root.name}: {e}", file=sys.stderr)
            failed += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
