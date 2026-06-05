"""Longitudinal comparison pairs — cross-epoch skill regression detection.

Analogous to the slow update mechanism in official SkillOpt: re-evaluates
the same cases with prev_epoch_skill and curr_epoch_skill, classifies each
pair into one of four categories, and feeds the comparison to the optimizer.

Categories:
  regressed       — prev ✓ → curr ✗  (skill change introduced a regression)
  improved        — prev ✗ → curr ✓  (skill change fixed something)
  persistent_fail — prev ✗ → curr ✗  (still failing; needs different approach)
  stable_success  — prev ✓ → curr ✓  (both skills passed)

The optimizer uses this signal to write or refine the slow_update guidance
field inside the skill document — a protected region between markers:
  <!-- SLOW_UPDATE_START --> ... <!-- SLOW_UPDATE_END -->

This is the key mechanism that produces +23.5 point improvements in the
official SkillOpt paper: it prevents skill drift by identifying which
rule changes actually help vs which accidentally break things.
"""

from __future__ import annotations

from typing import Any, Callable

from .update import SLOW_UPDATE_START, SLOW_UPDATE_END


def build_comparison_pairs(
    prev_results: list[dict[str, Any]],
    curr_results: list[dict[str, Any]],
    *,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build comparison pairs from two sets of rollout results.

    Each result dict should have `score` and `label` fields.

    Returns list of {label, prev_score, curr_score, category, delta}.
    """
    if labels is None:
        labels = [r.get("label", f"case_{i}") for i, r in enumerate(prev_results)]

    pairs: list[dict[str, Any]] = []
    for i in range(min(len(prev_results), len(curr_results))):
        prev_score = prev_results[i].get("score", 0.0)
        curr_score = curr_results[i].get("score", 0.0)
        label = labels[i] if i < len(labels) else f"case_{i}"

        if prev_score >= 0.5 and curr_score < 0.5:
            category = "regressed"
        elif prev_score < 0.5 and curr_score >= 0.5:
            category = "improved"
        elif prev_score < 0.5 and curr_score < 0.5:
            category = "persistent_fail"
        else:
            category = "stable_success"

        pairs.append({
            "label": label,
            "prev_score": round(prev_score, 3),
            "curr_score": round(curr_score, 3),
            "category": category,
            "delta": round(curr_score - prev_score, 3),
            "prev_passed": prev_score >= 0.5,
            "curr_passed": curr_score >= 0.5,
        })

    return pairs


def format_comparison_context(pairs: list[dict[str, Any]]) -> str:
    """Format comparison pairs as text for the slow update optimizer prompt.

    Returns empty string when there are no pairs.
    """
    if not pairs:
        return ""

    n_regressed = sum(1 for p in pairs if p["category"] == "regressed")
    n_improved = sum(1 for p in pairs if p["category"] == "improved")
    n_persist = sum(1 for p in pairs if p["category"] == "persistent_fail")
    n_stable = sum(1 for p in pairs if p["category"] == "stable_success")

    lines = [
        f"## Longitudinal Comparison (prev epoch skill vs current epoch skill)",
        f"Same tasks evaluated with BOTH skill versions:",
        f"",
        f"  Regressed ({n_regressed}): prev PASS → curr FAIL — the skill change broke something",
        f"  Improved ({n_improved}): prev FAIL → curr PASS — the skill change fixed something",
        f"  Persistent ({n_persist}): prev FAIL → curr FAIL — still broken, needs different approach",
        f"  Stable ({n_stable}): prev PASS → curr PASS — skill working as before",
        f"",
    ]

    # Show regressions first (most actionable)
    if n_regressed > 0:
        lines.append("### Regressions (prev PASS → curr FAIL)")
        for p in pairs:
            if p["category"] == "regressed":
                lines.append(f"- {p['label']}: {p['prev_score']:.2f} → {p['curr_score']:.2f}")

    # Show improvements
    if n_improved > 0:
        lines.append("\n### Improvements (prev FAIL → curr PASS)")
        for p in pairs:
            if p["category"] == "improved":
                lines.append(f"- {p['label']}: {p['prev_score']:.2f} → {p['curr_score']:.2f}")

    # Show persistent failures
    if n_persist > 0:
        lines.append("\n### Persistent Failures (still FAIL)")
        for p in pairs:
            if p["category"] == "persistent_fail":
                lines.append(f"- {p['label']}: {p['prev_score']:.2f} → {p['curr_score']:.2f}")

    return "\n".join(lines)


_SLOW_UPDATE_SYSTEM = """You are an expert skill optimizer analyzing cross-epoch comparison data.
Your job is to update the slow-update guidance section of a skill document
based on observed regressions, improvements, and persistent failures.

The slow-update section is a protected part of the skill document that
accumulates lessons learned across epochs. It should contain:
1. Patterns that consistently cause regressions → DO NOT DO these
2. Patterns that consistently produce improvements → REINFORCE these
3. Persistent failures that need a fundamentally different approach

Return JSON:
{
  "slow_update_content": "the updated guidance text (replace the entire section)",
  "observations": ["key observations from the comparison data"],
  "action": "accept" | "reject"
}

Rules:
- Be concise (50-150 words for slow_update_content)
- Focus on actionable lessons, not abstract observations
- If no clear pattern exists, return minimal guidance and set action="reject"
- The guidance replaces ALL previous slow-update content — include what's still relevant"""


def run_slow_update(
    skill_content: str,
    prev_results: list[dict[str, Any]],
    curr_results: list[dict[str, Any]],
    *,
    optimizer: Callable[[str, str, dict | None], tuple[str, Any]],
    existing_guidance: str = "",
    max_tokens: int = 4096,
) -> dict[str, Any] | None:
    """Run the slow update analysis: build comparison pairs → optimizer writes guidance.

    Args:
        skill_content: current skill document
        prev_results: rollout results from previous epoch's skill
        curr_results: rollout results from current epoch's skill
        optimizer: (system, user, options) → (response, metadata)
        existing_guidance: existing slow-update content (to refine)
        max_tokens: max completion tokens for optimizer

    Returns:
        dict with {slow_update_content, observations, action} or None on failure
    """
    pairs = build_comparison_pairs(prev_results, curr_results)
    if not pairs:
        return None

    comparison_context = format_comparison_context(pairs)
    if not comparison_context:
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"{comparison_context}\n\n"
    )
    if existing_guidance:
        user += (
            f"## Current Slow-Update Guidance (Refine or Replace)\n"
            f"{existing_guidance}\n\n"
        )
    user += (
        "Write updated slow-update guidance based on the comparison above. "
        "Return JSON with `slow_update_content`."
    )

    try:
        response, _meta = optimizer(_SLOW_UPDATE_SYSTEM, user, {
            "max_tokens": max_tokens,
            "stage": "slow_update",
        })
    except Exception:
        return None

    if not response:
        return None

    return _parse_slow_update_response(response, existing_guidance)


def _parse_slow_update_response(response: str, fallback: str) -> dict[str, Any] | None:
    import json as _json, re

    text = response.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"slow_update_content"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = _json.loads(match.group(0))
            except _json.JSONDecodeError:
                return None
        else:
            return None

    content = str(data.get("slow_update_content", "")).strip()
    if not content:
        return None

    return {
        "slow_update_content": content,
        "observations": data.get("observations", []),
        "action": data.get("action", "accept"),
    }


def inject_empty_slow_update(skill: str) -> str:
    """Inject empty slow-update markers if not present."""
    if SLOW_UPDATE_START in skill:
        return skill
    return skill.rstrip() + f"\n\n{SLOW_UPDATE_START}\n{SLOW_UPDATE_END}\n"


def replace_slow_update_content(skill: str, new_content: str) -> str:
    """Replace the content between SLOW_UPDATE markers with new_content."""
    start = skill.find(SLOW_UPDATE_START)
    end = skill.find(SLOW_UPDATE_END)
    if start == -1 or end == -1:
        return inject_empty_slow_update(skill).replace(
            f"{SLOW_UPDATE_START}\n{SLOW_UPDATE_END}",
            f"{SLOW_UPDATE_START}\n{new_content.strip()}\n{SLOW_UPDATE_END}",
        )
    return (
        skill[: start + len(SLOW_UPDATE_START) + 1]
        + new_content.strip()
        + "\n"
        + skill[end:]
    )


def extract_slow_update_content(skill: str) -> str:
    """Extract content between SLOW_UPDATE markers. Returns '' if none."""
    start = skill.find(SLOW_UPDATE_START)
    end = skill.find(SLOW_UPDATE_END)
    if start == -1 or end == -1:
        return ""
    content_start = start + len(SLOW_UPDATE_START)
    return skill[content_start:end].strip()
