"""mm (vlm-run/mm) auto-install bootstrap.

`mm` is a high-performance multimodal context-management CLI from vlm-run.
It indexes directories (~60ms for 700 files), exposes Unix-style
commands (`find`, `cat`, `grep`, `wc`, `tree`) and a `cat -m accurate`
mode that captions images / summarizes PDFs / transcribes audio via an
OpenAI-compatible VLM endpoint.

Why tinyctx installs it
───────────────────────
Image-bearing turns force the proxy to escalate to frontier
(`image_prefer_frontier = True`) because the local 27B has no reliable
vision. If `mm` is available, the proxy can run
`mm cat <attachment> -m accurate --format json` to obtain a text
caption, splice it into `body.input`, and keep the turn on the local
backend — that's the cost win. See `tinyctx/multimodal_preprocess.py`
for the run-time integration; this module just installs the binary.

Install path
────────────
Only the upstream shell installer is used:

    curl -LsSf https://vlm-run.github.io/mm/install/install.sh | sh

It lands at `~/.local/bin/mm` (Linux/macOS). Skipped on Windows because
the installer is `.ps1`-based.

Disable
───────
- `TINYCTX_MM_DISABLE=1` — bypass everything
- `TINYCTX_MM_BIN=/path/to/mm` — force a specific binary

CLI:
    python -m tinyctx.mm_bootstrap status
    python -m tinyctx.mm_bootstrap install
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


MM_BIN = "mm"
MM_INSTALL_URL = "https://vlm-run.github.io/mm/install/install.sh"

TINYCTX_HOME = Path(os.environ.get("TINYCTX_HOME", str(Path.home() / ".tinyctx")))
LOG_FILE = TINYCTX_HOME / "logs" / "mm-bootstrap.log"


def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def _which(cmd: str) -> str:
    """PATH + common user-local dirs the official installer writes to."""
    found = shutil.which(cmd)
    if found:
        return found
    for d in (
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.cargo/bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ):
        candidate = os.path.join(d, cmd)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


@dataclass
class MmState:
    mm_present: bool
    mm_path: str
    version: str
    disabled: bool


def detect_state() -> MmState:
    disabled = os.environ.get("TINYCTX_MM_DISABLE") == "1"
    forced = os.environ.get("TINYCTX_MM_BIN") or ""
    path = forced if forced and os.path.isfile(forced) else _which(MM_BIN)
    version = ""
    if path:
        try:
            proc = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=5)
            version = (proc.stdout or proc.stderr or "").strip().splitlines()[0][:80]
        except (OSError, subprocess.TimeoutExpired):
            version = ""
    return MmState(
        mm_present=bool(path),
        mm_path=path,
        version=version,
        disabled=disabled,
    )


def install() -> dict[str, object]:
    """Run the upstream shell installer. Returns a structured result."""
    if os.environ.get("TINYCTX_MM_DISABLE") == "1":
        _log("install skipped: TINYCTX_MM_DISABLE=1")
        return {"installed": False, "skipped": True, "reason": "disabled"}

    existing = _which(MM_BIN)
    if existing:
        _log(f"install skipped: already present at {existing}")
        return {"installed": False, "skipped": True, "path": existing}

    if sys.platform == "win32":
        _log("install skipped: windows uses install.ps1, not supported here")
        return {"installed": False, "skipped": True,
                "reason": "windows: use 'irm install.ps1 | iex' manually"}

    curl = _which("curl")
    sh = _which("sh") or "/bin/sh"
    if not curl:
        _log("install failed: curl not on PATH")
        return {"installed": False, "error": "curl not on PATH"}

    _log(f"installing mm via {MM_INSTALL_URL}")
    try:
        proc = subprocess.run(
            f"{curl} -LsSf {MM_INSTALL_URL} | {sh}",
            shell=True, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        _log("install timed out after 5min")
        return {"installed": False, "error": "install timed out"}
    except OSError as e:
        _log(f"install OSError: {e}")
        return {"installed": False, "error": f"install OSError: {e}"}

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        _log(f"install rc={proc.returncode}: {tail}")
        return {"installed": False, "error": f"rc={proc.returncode}: {tail}"}

    new_path = _which(MM_BIN)
    if not new_path:
        _log("install rc=0 but mm not on PATH afterwards")
        return {"installed": False,
                "error": "installer rc=0 but binary not on PATH"}
    _log(f"install ok: {new_path}")
    return {"installed": True, "path": new_path}


def _main() -> int:
    p = argparse.ArgumentParser(prog="tinyctx.mm_bootstrap")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("install")
    args = p.parse_args()
    if args.cmd == "status":
        import json as _j
        print(_j.dumps(asdict(detect_state()), indent=2))
        return 0
    if args.cmd == "install":
        import json as _j
        print(_j.dumps(install(), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
