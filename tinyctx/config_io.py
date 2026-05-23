"""Read, patch, and atomically save tinyctx TOML config files."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

try:  # py>=3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.9/3.10 fallback
    import tomli as tomllib  # type: ignore

from .config import load_config


_SECTION_RE = re.compile(r"^\s*\[([A-Za-z0-9_.-]+)\]\s*(?:#.*)?$")
_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*=")


def config_path() -> Path:
    return Path(os.environ.get(
        "TINYCTX_CONFIG",
        str(Path.home() / ".tinyctx" / "config.toml"),
    )).expanduser()


def read_config_text(path: Path | None = None) -> dict[str, Any]:
    target = path or config_path()
    raw = target.read_text(encoding="utf-8") if target.is_file() else ""
    parsed = tomllib.loads(raw) if raw.strip() else {}
    return {
        "path": str(target),
        "exists": target.is_file(),
        "raw": raw,
        "parsed": parsed,
    }


def effective_config() -> dict[str, Any]:
    cfg = load_config()
    return {
        "server": {
            "host": cfg.host,
            "port": cfg.port,
            "verbose": cfg.verbose,
        },
        "routing": {
            "force_route": cfg.force_route,
            "redirect_compaction_to_local": cfg.redirect_compaction_to_local,
            "sanitize_encrypted_content": cfg.sanitize_encrypted_content,
            "self_classify_escalates_to_frontier": cfg.self_classify_escalates_to_frontier,
            "escalate_input_tokens": cfg.escalate_input_tokens,
            "escalate_turn_count": cfg.escalate_turn_count,
            "escalate_on_error_streak": cfg.escalate_on_error_streak,
            "self_classify_escalates_to_frontier": cfg.self_classify_escalates_to_frontier,
        },
        "local": _backend_to_dict(cfg.local),
        "frontier": _backend_to_dict(cfg.frontier),
    }


def env_overrides() -> dict[str, Any]:
    mapping = {
        "local.base_url": "TINYCTX_LOCAL_BASE_URL",
        "local.model": "TINYCTX_LOCAL_MODEL",
        "local.wire_api": "TINYCTX_LOCAL_WIRE_API",
        "frontier.base_url": "TINYCTX_FRONTIER_BASE_URL",
        "frontier.model": "TINYCTX_FRONTIER_MODEL",
        "frontier.wire_api": "TINYCTX_FRONTIER_WIRE_API",
        "routing.force_route": "TINYCTX_FORCE_ROUTE",
        "routing.self_classify_escalates_to_frontier": "TINYCTX_SELF_CLASSIFY_ESCALATES_TO_FRONTIER",
        "server.verbose": "TINYCTX_VERBOSE",
    }
    return {
        field: {
            "env": env,
            "set": os.environ.get(env) is not None,
            "preview": _secret_preview(os.environ.get(env)),
        }
        for field, env in mapping.items()
    }


def merge_sections_into_toml(raw: str, sections: dict[str, Any]) -> str:
    clean = _clean_sections(sections)
    lines = raw.splitlines()

    for section, values in clean.items():
        lines = _merge_one_section(lines, section, values)

    return "\n".join(lines).rstrip() + "\n"


def save_config_text(raw: str, *, path: Path | None = None) -> dict[str, str | None]:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if target.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = target.with_suffix(target.suffix + f".bak-{stamp}")
        shutil.copy2(target, backup_path)

    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(raw)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "path": str(target),
        "backup_path": str(backup_path) if backup_path else None,
    }


def save_sections(sections: dict[str, Any], *, path: Path | None = None) -> dict[str, str | None]:
    current = read_config_text(path)
    merged = merge_sections_into_toml(current["raw"], sections)
    return save_config_text(merged, path=path or Path(current["path"]))


def _backend_to_dict(backend: Any) -> dict[str, Any]:
    return {
        "base_url": backend.base_url,
        "wire_api": backend.wire_api,
        "model": backend.model,
        "api_key_env": backend.api_key_env,
        "forward_authorization": backend.forward_authorization,
        "context_window": backend.context_window,
        "timeout_s": backend.timeout_s,
        "headers": dict(backend.headers or {}),
        "strip_tools": backend.strip_tools,
        "lmcache_passthrough": backend.lmcache_passthrough,
    }


def _clean_sections(sections: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for section, values in (sections or {}).items():
        if not isinstance(section, str) or not isinstance(values, dict):
            continue
        cleaned = {
            str(k): v for k, v in values.items()
            if isinstance(k, str) and v is not None
        }
        if cleaned:
            out[section] = cleaned
    return out


def _merge_one_section(lines: list[str], section: str, values: dict[str, Any]) -> list[str]:
    start, end = _section_range(lines, section)
    value_lines = {k: f"{k} = {_toml_value(v)}" for k, v in values.items()}
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"[{section}]")
        lines.extend(value_lines.values())
        return lines

    seen: set[str] = set()
    out = list(lines)
    for idx in range(start + 1, end):
        match = _KEY_RE.match(out[idx])
        if not match:
            continue
        key = match.group(1)
        if key in value_lines:
            out[idx] = value_lines[key]
            seen.add(key)

    missing = [line for key, line in value_lines.items() if key not in seen]
    if missing:
        insert_at = end
        out[insert_at:insert_at] = missing
    return out


def _section_range(lines: list[str], section: str) -> tuple[int | None, int]:
    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        match = _SECTION_RE.match(line)
        if not match:
            continue
        if match.group(1) == section:
            start = idx
            end = len(lines)
            continue
        if start is not None:
            end = idx
            break
    return start, end


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        parts = [f"{k} = {_toml_value(v)}" for k, v in value.items()]
        return "{ " + ", ".join(parts) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _secret_preview(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return value[:2] + "…" + value[-4:]
