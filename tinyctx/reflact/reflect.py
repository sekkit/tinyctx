"""ReflACT Reflect stage — trajectory analysis and edit-patch generation.

Analogous to gradient computation in neural network training: analyzes
minibatch rollout trajectories to produce skill-edit patches.

Two modes:
  1. analyze_failures:  group failure trajectories → optimizer proposes fixes
  2. analyze_successes: group success trajectories → optimizer proposes reinforcements

Context signals (fed into optimizer prompts):
  - step_buffer_context: past rejected edits + failure patterns from this epoch
  - meta_skill_context:  cross-epoch optimizer guidance (from meta_skill.py)

Trajectories come from tinyctx's trajectory.py JSONL format.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

OptimizerFn = Callable[[str, str, dict[str, Any] | None], tuple[str, Any]]

# ── trajectory formatting ─────────────────────────────────────────────

_MAX_TRAJ_CHARS = 12_000


def _clip(text: Any, limit: int = 500) -> str:
    if text is None:
        return ""
    return str(text)[:limit]


def fmt_trajectory(
    events: Sequence[dict[str, Any]],
    max_chars: int = _MAX_TRAJ_CHARS,
) -> str:
    """Format trajectory events into analyst-readable compact text."""
    lines: list[str] = []
    for item in events:
        phase = str(item.get("phase", "?"))[:20]
        event = str(item.get("event", "?"))[:60]
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
        artifacts = item.get("artifacts", {}) if isinstance(item.get("artifacts"), dict) else {}

        parts = [f"[{phase}] {event}"]
        if metrics:
            metric_str = " ".join(
                f"{k}={v}" for k, v in metrics.items()
                if isinstance(v, (int, float, str, bool))
            )
            if metric_str:
                parts.append(f"({metric_str})")
        if artifacts:
            key_list = list(artifacts.keys())[:5]
            parts.append(f"artifacts={key_list}")
        lines.append(" ".join(parts))

    text = "\n".join(lines)
    if len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = head + "\n\n... [middle truncated] ...\n\n" + tail
    return text


def fmt_rollout_result(
    events: Sequence[dict[str, Any]],
    score: float | None = None,
    label: str = "",
) -> str:
    header = f"## Rollout: {label}" if label else "## Rollout"
    if score is not None:
        header += f" (score={score:.3f})"
    traj_text = fmt_trajectory(events)
    return f"{header}\n{traj_text}"


# ── prompt templates ───────────────────────────────────────────────────

_ERROR_ANALYST_SYSTEM = """You are an expert skill optimizer. Your job is to analyze agent
FAILURE trajectories and propose precise, minimal edits to the skill
document that would prevent these failures from recurring.

## Edit Operations
- "replace": change an existing rule that caused or contributed to the failure.
  The `target` MUST be the EXACT text to replace, copied verbatim from the
  current skill document above.
- "append": add a new rule where one was completely missing.
- "delete": remove a rule that backfired (caused harm or wasted effort).
- "insert_after": add a new rule immediately after a specific existing rule.

## Constraints
1. Every `target` MUST be an exact substring of the current skill document.
   If you can't find the exact target text in the skill, use "append" instead.
2. `content` must be specific and actionable — NOT vague like "be more careful".
   It should read like a concrete rule the agent can follow.
3. Prefer "replace" over "append" when an existing rule is close to correct
   but missing detail or precision.
4. Every edit MUST cite specific evidence from the trajectories.
5. Each edit MUST address a distinct failure pattern — don't propose multiple
   edits for the same root cause.
6. If the trajectories don't suggest any skill improvement, return {"edits": []}.
7. Limit to at most 5 edits — focus on the highest-impact ones.

## Output Format
Return a JSON object:
{
  "reasoning": "concise analysis of failure patterns across the batch",
  "edits": [
    {
      "op": "replace",
      "target": "EXACT text from the current skill",
      "content": "replacement text",
      "reasoning": "which trajectory failures this edit prevents and why"
    }
  ]
}"""

_SUCCESS_ANALYST_SYSTEM = """You are an expert skill optimizer. Your job is to analyze agent
SUCCESS trajectories and propose edits that reinforce and generalize
the successful patterns so they apply to future scenarios.

## Edit Operations for Success
- "replace": refine a rule that was partially followed — the rule works but
  could be made more precise or general.
- "append": add a reinforcement note about a pattern that consistently works.
- "insert_after": add a reinforcement immediately after a related rule.

## Constraints
1. Every `target` MUST be an exact substring of the current skill document.
2. `content` should reinforce a specific successful pattern with precision.
3. Do NOT propose edits that merely praise the agent ("keep up the good work").
   Only propose edits when the success suggests a concrete rule improvement.
4. If the success is just "agent followed existing rules correctly" with no
   new pattern discovered, return {"edits": []}.
5. Limit to at most 3 edits from successes (failures are higher priority).

## Output Format
Return a JSON object:
{
  "reasoning": "what success patterns were observed and why they generalize",
  "edits": [
    {
      "op": "replace",
      "target": "EXACT text from the current skill",
      "content": "refined text that generalizes the success",
      "reasoning": "why this change makes the success pattern a permanent rule"
    }
  ]
}"""


def _build_user_prompt(
    skill_content: str,
    trajectories_block: str,
    *,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
) -> str:
    """Build the user prompt for the optimizer, including context signals."""
    parts = []

    if meta_skill_context:
        parts.append(meta_skill_context)

    parts.append("## Current Skill Document\n")
    parts.append(skill_content)
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(trajectories_block)

    if step_buffer_context:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append("## Previous Steps In This Epoch (rejected edits + patterns)")
        parts.append(step_buffer_context)
        parts.append("")
        parts.append("Avoid repeating edits that were already rejected. "
                      "If a pattern keeps failing despite edits, try a fundamentally "
                      "different approach this time.")

    parts.append("")
    parts.append("Analyze the trajectories above and propose edits. Return JSON.")

    return "\n".join(parts)


# ── optimizer call ────────────────────────────────────────────────────


def _call_optimizer(
    system: str,
    user: str,
    *,
    optimizer: OptimizerFn,
    max_tokens: int = 4096,
    stage: str = "reflect",
) -> tuple[str | None, str]:
    try:
        response, _meta = optimizer(system, user, {
            "max_tokens": max_tokens,
            "stage": stage,
        })
        return response, ""
    except Exception as e:
        return None, str(e)


def _extract_edits_from_response(response: str) -> list[dict[str, Any]]:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1] if text.count("```") >= 2 else text
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"edits"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    edits = data.get("edits", [])
    if not isinstance(edits, list):
        return []
    return [e for e in edits if isinstance(e, dict)]


# ── public API ─────────────────────────────────────────────────────────


def analyze_failures(
    skill_content: str,
    failed_rollouts: list[dict[str, Any]],
    *,
    optimizer: OptimizerFn,
    max_tokens: int = 4096,
    minibatch_size: int = 4,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
) -> dict[str, Any]:
    """Analyze failure rollouts in minibatches and propose corrective edits.

    Args:
        skill_content: current skill document
        failed_rollouts: list of {events, score, label}
        optimizer: (system, user, options) → (response, metadata)
        max_tokens: max completion tokens
        minibatch_size: trajectories grouped per optimizer call
        step_buffer_context: formatted text of previously rejected edits
        meta_skill_context: cross-epoch optimizer guidance

    Returns:
        {patches, source_type, batch_size, n_edits}
    """
    all_patches: list[dict[str, Any]] = []

    for i in range(0, len(failed_rollouts), minibatch_size):
        batch = failed_rollouts[i: i + minibatch_size]
        block_parts = []
        for j, rollout in enumerate(batch):
            events = rollout.get("events", [])
            score = rollout.get("score")
            label = rollout.get("label", f"failure_{i+j}")
            block_parts.append(fmt_rollout_result(events, score=score, label=label))

        trajectories_block = "\n\n".join(block_parts)
        user = _build_user_prompt(
            skill_content, trajectories_block,
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
        )

        response, err = _call_optimizer(
            _ERROR_ANALYST_SYSTEM, user,
            optimizer=optimizer, max_tokens=max_tokens, stage="reflect_failure",
        )
        if err:
            all_patches.append({"error": err, "edits": []})
            continue

        edits = _extract_edits_from_response(response or "")
        all_patches.append({
            "response": response,
            "edits": edits,
            "batch_size": len(batch),
            "source_type": "failure",
        })

    total_edits = sum(len(p.get("edits", [])) for p in all_patches)
    return {
        "patches": all_patches,
        "source_type": "failure",
        "batch_size": len(failed_rollouts),
        "n_edits": total_edits,
    }


def analyze_successes(
    skill_content: str,
    successful_rollouts: list[dict[str, Any]],
    *,
    optimizer: OptimizerFn,
    max_tokens: int = 4096,
    minibatch_size: int = 4,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
) -> dict[str, Any]:
    """Analyze success rollouts and propose reinforcement edits.

    Success edits are lower priority than failure edits — the aggregate
    stage applies failure patches first.

    Returns:
        {patches, source_type, batch_size, n_edits}
    """
    all_patches: list[dict[str, Any]] = []

    for i in range(0, len(successful_rollouts), minibatch_size):
        batch = successful_rollouts[i: i + minibatch_size]
        block_parts = []
        for j, rollout in enumerate(batch):
            events = rollout.get("events", [])
            label = rollout.get("label", f"success_{i+j}")
            block_parts.append(fmt_rollout_result(events, score=1.0, label=label))

        trajectories_block = "\n\n".join(block_parts)
        user = _build_user_prompt(
            skill_content, trajectories_block,
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
        )

        response, err = _call_optimizer(
            _SUCCESS_ANALYST_SYSTEM, user,
            optimizer=optimizer, max_tokens=max_tokens, stage="reflect_success",
        )
        if err:
            all_patches.append({"error": err, "edits": []})
            continue

        edits = _extract_edits_from_response(response or "")
        all_patches.append({
            "response": response,
            "edits": edits,
            "batch_size": len(batch),
            "source_type": "success",
        })

    total_edits = sum(len(p.get("edits", [])) for p in all_patches)
    return {
        "patches": all_patches,
        "source_type": "success",
        "batch_size": len(successful_rollouts),
        "n_edits": total_edits,
    }


def merge_edits(
    patches: list[dict[str, Any]],
    *,
    max_edits: int | None = None,
) -> list[dict[str, Any]]:
    """Simple dedup merge (fallback when hierarchical aggregate is skipped).

    Deduplicates by (op, target). Failure patches before success patches.
    """
    all_edits: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    failure_patches = [p for p in patches if p.get("source_type") != "success"]
    success_patches = [p for p in patches if p.get("source_type") == "success"]

    for patch in failure_patches + success_patches:
        source_type = patch.get("source_type", "failure")
        for edit in patch.get("edits", []):
            key = (str(edit.get("op", "")), str(edit.get("target", "")))
            if key in seen:
                continue
            seen.add(key)
            edit_copy = dict(edit)
            edit_copy["_source_type"] = source_type
            all_edits.append(edit_copy)

    if max_edits is not None and len(all_edits) > max_edits:
        all_edits = all_edits[:max_edits]

    return all_edits
