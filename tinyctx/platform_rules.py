"""Platform-specific guidance injection.

Injects a short block into body.instructions with the current platform
(Darwin/Linux/Windows) so models operating on foreign OSes get
platform-appropriate shell commands, file paths, and tool choices.

Cached — derived once at import time from platform.system().
"""

from __future__ import annotations

import platform
from typing import Any

_SYSTEM = platform.system()

_TEMPLATES: dict[str, str] = {
    "Darwin": (
        "\n## Platform\n"
        "You are running on macOS (Darwin). "
        "Use zsh as the default shell. "
        "Prefer `brew` for package management. "
        "File paths use `/` separators. "
        "Use `open` to open files and applications.\n"
    ),
    "Linux": (
        "\n## Platform\n"
        "You are running on Linux. "
        "Use bash as the default shell. "
        "Prefer `apt` or system-native package managers. "
        "File paths use `/` separators. "
        "Use `xdg-open` to open files and URLs.\n"
    ),
    "Windows": (
        "\n## Platform\n"
        "You are running on Windows. "
        "Use PowerShell as the default shell. "
        "Prefer `winget` or `choco` for package management. "
        "File paths use `\\` separators. "
        "Use `start` to open files and URLs.\n"
    ),
}

_PLATFORM_BLOCK = _TEMPLATES.get(_SYSTEM, "")


def template_chars() -> int:
    """Return the character count of the platform block (for trace stats)."""
    return len(_PLATFORM_BLOCK)


def inject_into_body(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Append platform guidance to body.instructions. Returns (body, injected)."""
    if not _PLATFORM_BLOCK:
        return body, False

    inst = body.get("instructions") or ""
    if _PLATFORM_BLOCK in str(inst):
        return body, False

    out = dict(body)
    out["instructions"] = str(inst) + _PLATFORM_BLOCK
    return out, True
