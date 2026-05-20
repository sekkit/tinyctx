"""Dynamic Skill generation and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


@dataclass
class DynamicSkill:
    name: str
    scope: str
    use_when: str
    steps: List[str]
    do_not: List[str]
    verification: List[str]
    source: str
    content_hash: str


Planner = Callable[[str, str], Any]


def build_dynamic_skill(
    task: str,
    gap: str,
    planner: Optional[Planner] = None,
) -> DynamicSkill:
    """Build a current-turn Dynamic Skill, falling back safely on planner failure."""

    if planner is not None:
        try:
            planned = _coerce_skill(planner(task, gap), default_source="planner")
            planned.content_hash = _content_hash(planned)
            if validate_dynamic_skill(planned)["ok"]:
                return planned
        except Exception:
            pass

    fallback = DynamicSkill(
        name="Conservative Dynamic Skill",
        scope="current task only",
        use_when=(
            "Use only for the current task: "
            f"{_brief(task)}. It covers this gap: {_brief(gap)}."
        ),
        steps=[
            "Restate the current task and allowed scope before acting.",
            "Use only available local tools and existing project context.",
            "Make the smallest verifiable change that addresses the task.",
        ],
        do_not=[
            "Do not change unrelated files or broaden the task.",
            "Do not override higher-priority instructions.",
            "Do not access sensitive credentials or request elevated permissions.",
            "Do not install or download arbitrary external software.",
        ],
        verification=[
            "Run the most focused available check for the changed behavior.",
        ],
        source="fallback",
        content_hash="",
    )
    fallback.content_hash = _content_hash(fallback)
    return fallback


def validate_dynamic_skill(skill: DynamicSkill) -> Dict[str, Any]:
    """Validate a Dynamic Skill against instruction and safety constraints."""

    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(skill, DynamicSkill):
        return {
            "ok": False,
            "errors": ["skill must be a DynamicSkill instance"],
            "warnings": [],
        }

    for field_name in ("name", "scope", "use_when", "source"):
        if not _clean(getattr(skill, field_name, "")):
            errors.append(f"{field_name} is required")

    for field_name in ("steps", "do_not", "verification"):
        value = getattr(skill, field_name, None)
        if not isinstance(value, list):
            errors.append(f"{field_name} must be a list")
        elif not value:
            errors.append(f"{field_name} is required")
        elif any(not _clean(item) for item in value):
            errors.append(f"{field_name} contains an empty item")

    if "current task only" not in _clean(skill.scope).lower():
        errors.append("scope must be current task only")

    chunks = _instruction_chunks(skill)
    if _has_instruction_hierarchy_bypass(chunks):
        errors.append("instruction hierarchy bypass is not allowed")
    if _has_secret_exfiltration(chunks):
        errors.append("secrets access or exfiltration is not allowed")
    if _has_permission_escalation(chunks):
        errors.append("permission escalation is not allowed")
    if _has_external_install(chunks):
        errors.append("arbitrary external install or download is not allowed")
    if _has_security_check_disable(chunks):
        errors.append("disabling safety or security checks is not allowed")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def render_dynamic_skill(skill: DynamicSkill, max_chars: int = 2000) -> str:
    """Render a bounded inline playbook for injection into the current request."""

    if max_chars <= 0:
        return ""

    lines = [
        f"## tinyctx Dynamic Skill: {_clean(skill.name)}",
        "",
        "Scope: current task only.",
        f"Use when: {_clean(skill.use_when)}",
        "Do:",
        *_bullets(skill.steps),
        "Do not:",
        *_bullets(skill.do_not),
        "Verify:",
        *_bullets(skill.verification),
        f"Source: {_clean(skill.source)}",
        f"Hash: {_clean(skill.content_hash)}",
    ]
    rendered = "\n".join(lines).strip()
    return _truncate(rendered, max_chars)


def _coerce_skill(value: Any, default_source: str) -> DynamicSkill:
    if isinstance(value, DynamicSkill):
        return DynamicSkill(
            name=_clean(value.name),
            scope=_clean(value.scope) or "current task only",
            use_when=_clean(value.use_when),
            steps=_string_list(value.steps),
            do_not=_string_list(value.do_not),
            verification=_string_list(value.verification),
            source=_clean(value.source) or default_source,
            content_hash="",
        )

    if not isinstance(value, Mapping):
        raise TypeError("planner must return a mapping or DynamicSkill")

    return DynamicSkill(
        name=_clean(value.get("name")) or "Generated Dynamic Skill",
        scope=_clean(value.get("scope")) or "current task only",
        use_when=_clean(value.get("use_when")),
        steps=_string_list(value.get("steps")),
        do_not=_string_list(value.get("do_not")),
        verification=_string_list(value.get("verification")),
        source=_clean(value.get("source")) or default_source,
        content_hash="",
    )


def _content_hash(skill: DynamicSkill) -> str:
    payload = asdict(skill)
    payload["content_hash"] = ""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _instruction_chunks(skill: DynamicSkill) -> List[str]:
    chunks: List[str] = [
        _clean(skill.name),
        _clean(skill.scope),
        _clean(skill.use_when),
    ]
    chunks.extend(_string_list(skill.steps))
    chunks.extend(_string_list(skill.verification))
    return [chunk for chunk in chunks if chunk and not _is_prohibition(chunk)]


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean(value)] if _clean(value) else []
    if isinstance(value, Iterable):
        return [_clean(item) for item in value if _clean(item)]
    return [_clean(value)] if _clean(value) else []


def _bullets(items: Iterable[str]) -> List[str]:
    return [f"- {_clean(item)}" for item in items if _clean(item)]


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _brief(value: Any, limit: int = 160) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _is_prohibition(text: str) -> bool:
    lowered = _clean(text).lower()
    return lowered.startswith(("do not ", "don't ", "never ", "avoid "))


def _has_instruction_hierarchy_bypass(chunks: Iterable[str]) -> bool:
    patterns = [
        r"\b(ignore|override|bypass|disregard)\b.*\b(system|developer|agents?)\b",
        r"\b(system|developer|agents?)\b.*\b(instruction|message|rule)s?\b.*\b(ignore|override|bypass|disregard)\b",
        r"\bhigher[- ]priority\b.*\b(ignore|override|bypass|disregard)\b",
    ]
    return _matches_any(chunks, patterns)


def _has_secret_exfiltration(chunks: Iterable[str]) -> bool:
    secret_terms = r"\b(secret|secrets|api[-_ ]?key|token|password|credential|credentials|\.env|ssh[-_ ]?key)\b"
    verbs = r"\b(read|print|dump|leak|exfiltrate|expose|upload|send|collect|harvest)\b"
    return _matches_any(chunks, [rf"{verbs}.*{secret_terms}", rf"{secret_terms}.*{verbs}"])


def _has_permission_escalation(chunks: Iterable[str]) -> bool:
    patterns = [
        r"\bsudo\b",
        r"\brun as administrator\b",
        r"\broot access\b",
        r"\bescalat\w*\b.*\b(permission|privilege|privileges)\b",
        r"\bchmod\s+777\b",
        r"\bset-executionpolicy\s+unrestricted\b",
    ]
    return _matches_any(chunks, patterns)


def _has_external_install(chunks: Iterable[str]) -> bool:
    patterns = [
        r"\bdownload\b.*\binstall\b.*\b(external|internet|arbitrary|any)\b",
        r"\binstall\b.*\b(downloaded|external|internet|arbitrary|any)\b",
        r"\b(curl|wget)\b.*\|\s*(bash|sh|powershell|pwsh)\b",
        r"\bpipe\b.*\b(curl|wget)\b.*\b(bash|sh|powershell|pwsh)\b",
    ]
    return _matches_any(chunks, patterns)


def _has_security_check_disable(chunks: Iterable[str]) -> bool:
    patterns = [
        r"\b(disable|turn off|bypass|skip)\b.*\b(security|safety)\b.*\b(check|checks|guard|guards|validation)\b",
        r"\b(security|safety)\b.*\b(check|checks|guard|guards|validation)\b.*\b(disable|bypass|skip)\b",
    ]
    return _matches_any(chunks, patterns)


def _matches_any(chunks: Iterable[str], patterns: Iterable[str]) -> bool:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for chunk in chunks:
        text = _clean(chunk)
        if any(pattern.search(text) for pattern in compiled):
            return True
    return False


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    suffix = "\n…"
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix
