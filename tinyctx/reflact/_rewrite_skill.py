"""Full skill rewrite — optimizer regenerates the entire skill document.

Instead of applying patches, the optimizer receives the current skill +
a set of revise_suggestions and produces a completely rewritten skill.

This mode is useful when:
  - The skill has been patched many times and may have accumulated cruft
  - A fundamental restructuring is needed
  - The patch budget is too small for all needed changes
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

OptimizerFn = Callable[[str, str, dict[str, Any] | None], tuple[str, Any]]

_REWRITE_SYSTEM = """You are an expert skill writer. You will receive a current agent skill
document and a set of revise suggestions extracted from trajectory analysis.

Your job is to rewrite the ENTIRE skill document so it integrates all
the selected suggestions cleanly. The output should be a complete, coherent
skill document — not a diff, not a set of patches.

Rules:
1. Preserve the overall structure and section hierarchy of the original.
2. Integrate ALL suggestions — if a suggestion conflicts with existing rules,
   resolve the conflict in favor of the suggestion (it is evidence-backed).
3. Remove rules that are contradicted by the suggestions.
4. Keep the document concise — do not add filler.
5. Preserve the original title and section headings unless a suggestion
   explicitly says to change them.
6. If a suggestion says "append", place the content in the most relevant
   existing section. If no section fits, add it to the end.

Return JSON:
{
  "new_skill": "the complete rewritten skill document",
  "change_summary": ["bullet list of changes made"],
  "title": "optional new title if changed"
}"""


def rewrite_skill_from_suggestions(
    skill_content: str,
    patch: dict[str, Any],
    *,
    optimizer: OptimizerFn,
    max_tokens: int = 64000,
    env: str | None = None,
) -> dict[str, Any] | None:
    """Rewrite the entire skill document incorporating selected suggestions.

    Args:
        skill_content: current skill document
        patch: dict with "edits" list (revise suggestions)
        optimizer: (system, user, options) → (response, metadata)
        max_tokens: max completion tokens (big: need room for full skill)
        env: optional environment name for prompt customization

    Returns:
        dict with {new_skill, change_summary, title} or None on failure
    """
    suggestions = patch.get("edits", [])
    if not suggestions:
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Selected Revise Suggestions ({len(suggestions)} total)\n"
        f"{json.dumps(suggestions, ensure_ascii=False, indent=2)}\n\n"
        "Rewrite the full skill document so it integrates these suggestions. "
        "Return the complete new skill in `new_skill`."
    )

    try:
        response, _meta = optimizer(_REWRITE_SYSTEM, user, {
            "max_tokens": max_tokens,
            "stage": "rewrite",
        })
    except Exception:
        return None

    if not response:
        return None

    text = response.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"new_skill"\s*:.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    new_skill = str(data.get("new_skill", "")).strip()
    if not new_skill:
        return None

    # Ensure trailing newline
    if not new_skill.endswith("\n"):
        new_skill += "\n"

    change_summary = data.get("change_summary", [])
    if not isinstance(change_summary, list):
        change_summary = []

    return {
        "new_skill": new_skill,
        "change_summary": change_summary,
        "title": str(data.get("title", "")),
    }
