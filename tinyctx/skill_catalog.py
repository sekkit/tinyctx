from __future__ import annotations

from copy import deepcopy
from typing import Any


_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "skills": {
        "cc-tdd": {
            "kind": "skill",
            "description": "Test-driven development workflow for fixes and features.",
            "task_types": ["coding", "debug"],
            "constraints": ["test-first"],
        },
        "cc-work": {
            "kind": "skill",
            "description": "General implementation workflow for scoped coding tasks.",
            "task_types": ["coding", "debug", "review", "docs"],
            "constraints": ["finish completely", "verify changes"],
        },
        "cc-design": {
            "kind": "skill",
            "description": "High-fidelity HTML/UI design and prototype workflow.",
            "task_types": ["design"],
            "constraints": ["visual verification"],
        },
        "huashu-design": {
            "kind": "skill",
            "description": "Advanced prototype, animation, and design exploration workflow.",
            "task_types": ["design"],
            "constraints": ["prototype-first"],
        },
        "cc-research": {
            "kind": "skill",
            "description": "Repository and external-context research orchestration.",
            "task_types": ["research"],
            "constraints": ["summarize findings"],
        },
    },
    "mcp": {
        "context-mode": {
            "kind": "mcp",
            "description": "Context-safe batch execution, indexing, and search tools.",
            "task_types": ["coding", "debug", "research", "docs", "review"],
            "constraints": ["use for large searches"],
        },
        "browser": {
            "kind": "mcp",
            "description": "In-app browser automation for local UI verification.",
            "task_types": ["design"],
            "constraints": ["verify local targets"],
        },
        "gitnexus": {
            "kind": "mcp",
            "description": "Git history and repository relationship analysis.",
            "task_types": ["coding", "debug", "review"],
            "constraints": ["inspect history when needed"],
        },
        "serena": {
            "kind": "mcp",
            "description": "Semantic code navigation and symbol-level editing support.",
            "task_types": ["coding", "debug", "review"],
            "constraints": ["prefer symbol-aware lookup"],
        },
        "advisor": {
            "kind": "mcp",
            "description": "Task guidance and recommendation support.",
            "task_types": ["coding", "debug", "research", "design", "review"],
            "constraints": ["use for routing advice"],
        },
    },
}


def default_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    return deepcopy(_CATALOG)


def summarize_catalog(catalog: dict[str, Any], max_chars: int = 1200) -> str:
    if max_chars <= 0:
        return ""

    lines: list[str] = []
    for group in ("skills", "mcp"):
        entries = catalog.get(group, {})
        if not isinstance(entries, dict):
            entries = {}
        parts: list[str] = []
        for name in sorted(entries):
            entry = entries.get(name) or {}
            task_types = entry.get("task_types") or []
            if task_types:
                parts.append(f"{name}({', '.join(task_types)})")
            else:
                parts.append(name)
        lines.append(f"{group}: " + ", ".join(parts))

    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    if max_chars <= 3:
        return summary[:max_chars]
    return summary[: max_chars - 3].rstrip() + "..."
