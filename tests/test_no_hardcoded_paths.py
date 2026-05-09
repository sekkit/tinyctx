"""Static scan: tracked source must not contain a per-developer home path.

Regression guard for the portability bug class:
  - tinyctx/advisor.py once embedded `/Users/sekkit/dev/tinyctx/.venv/...`
    in its docstring config example; copying that block verbatim broke
    every other developer's setup.
  - scripts/cm-hook-shim once defaulted CM_BIN to
    `/Users/sekkit/.local/node/bin/context-mode`; the binary doesn't
    exist for anyone but the original author.

Both fixes are runtime-resolved now (sys.executable / $HOME). This test
walks every git-tracked text file and fails if a `/Users/<name>/` or
`/home/<name>/` literal sneaks back in. The intent is that anyone who
clones tinyctx and runs `scripts/install.sh` has a working setup —
no manual path edits required.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

# Match `/Users/<name>` or `/home/<name>` followed by `/` or end-of-segment.
# We require an alphabetic first character on the username so we don't
# false-positive on `/home/.config` or absolute regex docs.
_PATTERN = re.compile(r"(?P<root>/Users|/home)/(?P<user>[A-Za-z][A-Za-z0-9._-]*)(?=[/'\"\s)\]:`]|$)")

# Names that are NOT real personal home dirs and may legitimately appear:
#   CI / sandbox accounts (legit at runtime, never a personal leak):
#     - `runner`     : GitHub Actions / GitLab CI default user
#     - `vscode`     : VS Code Dev Containers default user
#     - `codespace`  : GitHub Codespaces default user
#     - `linuxbrew`  : Homebrew on Linux installs under /home/linuxbrew
#   Conventional placeholders in docstrings and test fixtures (the
#   reader is meant to substitute their own username — these signal
#   "example", not a real path):
#     - `x`, `me`, `you`, `user`, `username`, `secret-username`
# Add to this list only with clear justification; the default is to fail.
_ALLOWED_USERNAMES = {
    "runner", "vscode", "codespace", "linuxbrew",
    "x", "me", "you", "user", "username", "secret-username",
}

# Paths (relative to repo root) that are themselves about explaining or
# checking this rule, so they can mention the patterns without tripping it.
_ALLOWED_PATHS = {
    "tests/test_no_hardcoded_paths.py",
}

# Filename suffixes / basenames we scan. Everything else (binaries,
# generated assets, lockfiles) is skipped — those rarely contain prose
# and tend to produce noise.
_SCANNED_SUFFIXES = {
    ".py", ".sh", ".bash", ".zsh", ".toml", ".json", ".jsonc",
    ".yaml", ".yml", ".md", ".txt", ".cfg", ".ini",
}
_SCANNED_BASENAMES = {
    "cm-hook-shim", "scout-session-start.sh", "tinyctx-up",
    "Makefile", "Dockerfile",
}


def _git_tracked_files(root: Path) -> list[Path]:
    """Return every tracked file in the repo. Falls back to walking the
    tree if git is unavailable (so the test still gives a useful signal
    in CI containers without git history)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root, check=True, capture_output=True, timeout=15,
        )
        names = [n for n in out.stdout.decode("utf-8", "replace").split("\0") if n]
        return [root / n for n in names]
    except (subprocess.SubprocessError, FileNotFoundError):
        # Best-effort fallback; skip noisy dirs.
        skip = {".git", ".venv", "node_modules", ".codex", ".claude",
                "tinyctx.egg-info", "graphify-out", ".codex-rollouts"}
        out2: list[Path] = []
        for p in root.rglob("*"):
            if any(part in skip for part in p.parts):
                continue
            if p.is_file():
                out2.append(p)
        return out2


def _should_scan(path: Path) -> bool:
    if path.suffix in _SCANNED_SUFFIXES:
        return True
    if path.name in _SCANNED_BASENAMES:
        return True
    return False


def _scan_file(path: Path, repo_root: Path) -> list[tuple[int, str, str]]:
    rel = str(path.relative_to(repo_root))
    if rel in _ALLOWED_PATHS:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _PATTERN.finditer(line):
            user = m.group("user")
            if user in _ALLOWED_USERNAMES:
                continue
            hits.append((lineno, m.group(0), line.strip()))
    return hits


def test_no_per_user_home_paths_in_tracked_source():
    files = _git_tracked_files(REPO_ROOT)
    assert files, "git ls-files returned nothing — test setup broken"

    failures: list[str] = []
    for f in files:
        if not _should_scan(f):
            continue
        for lineno, match, line in _scan_file(f, REPO_ROOT):
            failures.append(
                f"  {f.relative_to(REPO_ROOT)}:{lineno}  "
                f"matched {match!r}\n      → {line[:160]}"
            )

    if failures:
        msg = (
            "Found per-user home paths in tracked source. These break for "
            "any other developer who clones tinyctx. Replace them with "
            "runtime-resolved values (sys.executable, $HOME, "
            "Path.home(), command -v ...) or extend _ALLOWED_USERNAMES "
            "if the username is genuinely a CI/sandbox identity.\n\n"
            + "\n".join(failures)
        )
        pytest.fail(msg)


def test_pattern_matches_known_offenders():
    """Sanity: the regex would catch the historical bugs we just fixed."""
    historical = [
        '/Users/sekkit/dev/tinyctx/.venv/bin/python',
        '/Users/sekkit/.local/node/bin/context-mode',
        '/home/alice/dev/foo',
        'CM_BIN="${CM_BIN:-/Users/sekkit/.local/node/bin/context-mode}"',
    ]
    for s in historical:
        assert _PATTERN.search(s), f"pattern failed to flag: {s!r}"


def test_pattern_ignores_allowlist():
    """`runner`, `vscode`, etc. shouldn't trigger when used as the
    home-dir username — those are CI / dev-container conventions, not
    a personal account leaking into the repo."""
    benign = [
        "/home/runner/work/tinyctx/tinyctx",
        "/home/vscode/.config",
        "/Users/codespace/project",
    ]
    for s in benign:
        m = _PATTERN.search(s)
        # The pattern matches, but the username is on the allowlist.
        assert m is not None
        assert m.group("user") in _ALLOWED_USERNAMES


def test_pattern_doesnt_match_tilde_or_envvar():
    """Tilde and $HOME references are exactly the runtime-resolved form
    we want — they must not trigger the rule."""
    safe = [
        "~/dev/tinyctx",
        "$HOME/.local/bin/graphify",
        "${HOME}/.local/node/bin/context-mode",
        "Path.home() / '.tinyctx'",
    ]
    for s in safe:
        assert _PATTERN.search(s) is None, f"false positive on: {s!r}"
