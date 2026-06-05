"""ReflACT skill update — bounded add / delete / replace edits on a skill document.

Analogous to optimizer.step() in neural network training: applies a set of
ranked edits to the current skill document, producing an updated candidate.

Edit operations:
  - append         — add content at end of document (before slow-update region if present)
  - insert_after   — insert content after a target string; fallback = append
  - replace        — replace target string with content (first occurrence only)
  - delete         — remove target string (first occurrence only)

Protected regions (<!-- SLOW_UPDATE_START --> ... <!-- SLOW_UPDATE_END -->)
are immune to replace/delete edits. Append/insert_after place content before
the slow-update region when it exists.
"""

from __future__ import annotations

from typing import Any

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"


def _is_in_slow_update_region(skill: str, target: str) -> bool:
    """Return True if *target* text lies inside the protected slow-update region."""
    start_idx = skill.find(SLOW_UPDATE_START)
    end_idx = skill.find(SLOW_UPDATE_END)
    if start_idx == -1 or end_idx == -1:
        return False
    target_idx = skill.find(target)
    if target_idx == -1:
        return False
    region_end = end_idx + len(SLOW_UPDATE_END)
    return start_idx <= target_idx < region_end


def _strip_markers(text: str) -> str:
    """Remove slow-update markers from edit content to prevent duplication."""
    return (text.replace(SLOW_UPDATE_START, "")
                .replace(SLOW_UPDATE_END, ""))


def _edit_fields(edit: dict[str, Any]) -> tuple[str, str, str]:
    """Extract (op, content, target) from an edit dict with safe defaults."""
    op = str(edit.get("op", "")).strip().lower()
    content = _strip_markers(str(edit.get("content", "")).strip())
    target = str(edit.get("target", ""))
    return op, content, target


def apply_edit(skill: str, edit: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Apply a single edit to the skill document.

    Returns (new_skill, report_dict). The report records which edit was
    applied and why — used by the trainer for logging and rejection analysis.

    Args:
        skill: current skill document text
        edit: dict with keys {op, content, target} where:
            op: "append" | "insert_after" | "replace" | "delete"
            content: text to add (not used for delete)
            target: anchor string for insert_after / replace / delete
    """
    op, content, target = _edit_fields(edit)
    report: dict[str, Any] = {
        "op": op,
        "target": target[:200],
        "content_preview": content[:200],
        "status": "unknown",
    }

    # ── Guard: no replace/delete inside slow-update region ──
    if target and op in ("replace", "delete") and _is_in_slow_update_region(skill, target):
        report["status"] = "skipped_protected_slow_update_region"
        return skill, report

    # ── append ──
    if op == "append":
        su_start = skill.find(SLOW_UPDATE_START)
        if su_start != -1:
            before = skill[:su_start].rstrip()
            after = skill[su_start:]
            report["status"] = "applied_append_before_slow_update"
            return before + "\n\n" + content + "\n\n" + after, report
        report["status"] = "applied_append"
        return skill.rstrip() + "\n\n" + content + "\n", report

    # ── insert_after ──
    if op == "insert_after":
        if not target or target not in skill:
            # Fallback: append before slow-update or at end
            su_start = skill.find(SLOW_UPDATE_START)
            if su_start != -1:
                before = skill[:su_start].rstrip()
                after = skill[su_start:]
                report["status"] = "applied_insert_after_fallback_before_slow_update"
                return before + "\n\n" + content + "\n\n" + after, report
            report["status"] = "applied_insert_after_fallback_append"
            return skill.rstrip() + "\n\n" + content + "\n", report
        idx = skill.index(target) + len(target)
        newline = skill.find("\n", idx)
        insert_at = newline + 1 if newline != -1 else len(skill)
        report["status"] = "applied_insert_after"
        return skill[:insert_at] + "\n" + content + "\n" + skill[insert_at:], report

    # ── replace ──
    if op == "replace":
        if not target:
            report["status"] = "skipped_replace_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_replace_target_not_found"
            return skill, report
        report["status"] = "applied_replace"
        return skill.replace(target, content, 1), report

    # ── delete ──
    if op == "delete":
        if not target:
            report["status"] = "skipped_delete_missing_target"
            return skill, report
        if target not in skill:
            report["status"] = "skipped_delete_target_not_found"
            return skill, report
        report["status"] = "applied_delete"
        return skill.replace(target, "", 1), report

    report["status"] = f"unknown_op_{op}"
    return skill, report


def apply_patch(skill: str, edits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Apply a sequence of edits to the skill document.

    Edits are applied in order; each subsequent edit operates on the result
    of the previous one. Returns (new_skill, reports).
    """
    reports: list[dict[str, Any]] = []
    current = skill
    for edit in edits:
        current, report = apply_edit(current, edit)
        reports.append(report)
    return current, reports


def describe_edit(edit: dict[str, Any]) -> str:
    """Human-readable one-line summary of an edit for logging."""
    op = str(edit.get("op", "?"))
    target = str(edit.get("target", ""))[:60]
    content = str(edit.get("content", ""))[:80]
    if op in ("replace", "delete"):
        return f"[{op}] {target}"
    if op == "append":
        return f"[append] {content}"
    if op == "insert_after":
        return f"[insert_after] {target} ← {content}"
    return f"[{op}]"
