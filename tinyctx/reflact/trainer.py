"""ReflACT Trainer — skill optimization with longitudinal comparison.

Full feature parity with upstream SkillOpt (v0.1.0):
  - 6-stage step pipeline: Rollout→Reflect→Aggregate→Select→Update→Evaluate
  - Slow update (longitudinal pairs): epoch 2+ comparison of adjacent skills
  - Meta-skill: cross-epoch optimizer guidance document
  - Cosine / linear / constant / autonomous edit-budget scheduler
  - LLM-driven edit ranking (gradient clipping)
  - Hierarchical patch aggregation
  - Step buffer + meta-skill context fed to optimizer
  - Full-rewrite mode, autonomous LR mode

Default config matches official SkillOpt:
  epochs=4, accumulation=1, edit_budget=4, lr_scheduler=cosine,
  minibatch_size=8, merge_batch_size=8, use_slow_update=True,
  longitudinal_pair_policy=mixed, use_meta_skill=True
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .reflect import analyze_failures, analyze_successes, merge_edits
from .update import apply_patch, describe_edit

OptimizerFn = Callable[[str, str, dict[str, Any] | None], tuple[str, Any]]
EvaluatorFn = Callable[[str], list[dict[str, Any]]]  # skill → [{events, score, label}]


# ── types ──────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Configuration for a ReflACT training run.  Defaults match official SkillOpt."""

    # Paths
    skill_path: str = ""
    out_dir: str = ""

    # Training parameters (defaults: SkillOpt paper)
    epochs: int = 4
    accumulation: int = 1
    batch_size: int = 8
    edit_budget: int = 4
    min_edit_budget: int = 2
    minibatch_size: int = 8
    merge_batch_size: int = 8
    aggregate_workers: int = 8

    # Scheduler
    lr_scheduler: str = "cosine"

    # Update mode
    update_mode: str = "patch"

    # Slow update (longitudinal pairs) — official SkillOpt feature
    use_slow_update: bool = True
    slow_update_samples: int = 20
    longitudinal_pair_policy: str = "mixed"

    # Meta-skill (cross-epoch learning)
    use_meta_skill: bool = True
    meta_skill_interval: int = 1

    # Validation
    validation_split: float = 0.2
    gate_threshold: float = 0.0
    max_rejected_steps: int = 3

    # Optimizer
    optimizer_model: str = ""
    optimizer_max_tokens: int = 4096
    rewrite_max_tokens: int = 64000

    # Misc
    seed: int = 42
    verbose: bool = False


@dataclass
class TrainResult:
    initial_skill: str = ""
    best_skill: str = ""
    current_skill: str = ""
    best_score: float = 0.0
    best_step: int = 0
    epochs_completed: int = 0
    steps_completed: int = 0
    total_edits_applied: int = 0
    total_edits_rejected: int = 0
    elapsed_s: float = 0.0
    meta_skill: str = ""
    slow_updates_applied: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


# ── training loop ──────────────────────────────────────────────────────


def train_skill(
    cfg: TrainConfig,
    *,
    optimizer: OptimizerFn,
    rollout_fn: EvaluatorFn,
    eval_fn: EvaluatorFn | None = None,
) -> TrainResult:
    t_start = time.time()
    _eval = eval_fn or rollout_fn

    skill_path = Path(cfg.skill_path)
    if skill_path.is_file():
        initial_skill = skill_path.read_text(encoding="utf-8")
    else:
        initial_skill = "# Agent Skill Document\n\nNo initial rules defined.\n"
    current_skill = initial_skill
    best_skill = initial_skill
    best_score = 0.0
    best_step = 0

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .scheduler import build_scheduler
    scheduler = build_scheduler(
        mode=cfg.lr_scheduler,
        max_lr=cfg.edit_budget,
        min_lr=cfg.min_edit_budget,
        total_steps=cfg.epochs * cfg.accumulation,
    )

    if cfg.verbose:
        print(f"[reflact] initial skill: {skill_path} ({len(initial_skill)} chars)")
        print(f"[reflact] epochs={cfg.epochs} accumulation={cfg.accumulation} "
              f"edit_budget={cfg.edit_budget} scheduler={cfg.lr_scheduler} "
              f"slow_update={cfg.use_slow_update} meta_skill={cfg.use_meta_skill}")

    # ── Initial rollout ─────────────────────────────────────────────────
    all_rollouts = rollout_fn(current_skill)
    n_cases = len(all_rollouts)
    n_val = max(1, int(n_cases * cfg.validation_split))
    n_train = max(1, n_cases - n_val)

    if cfg.verbose:
        print(f"[reflact] cases: {n_cases} total, {n_train} train, {n_val} validation")

    val_rollouts = _eval(current_skill)
    current_score = _mean_score(val_rollouts)
    best_score = current_score

    if cfg.verbose:
        print(f"[reflact] baseline validation score: {current_score:.4f}")

    # ── State ──────────────────────────────────────────────────────────
    step_buffer: list[dict[str, Any]] = []
    active_meta_skill = ""
    history: list[dict[str, Any]] = []
    global_step = 0
    total_edits_applied = 0
    total_edits_rejected = 0
    consecutive_rejections = 0
    prev_epoch_skill = initial_skill
    slow_updates_applied = 0

    # ── Slow update: inject placeholder at epoch 1 ─────────────────────
    if cfg.use_slow_update and cfg.epochs >= 2:
        from .comparison import inject_empty_slow_update
        current_skill = inject_empty_slow_update(current_skill)
        if cfg.verbose:
            print("[reflact] slow_update: injected empty placeholder for epoch 1")

    for epoch in range(1, cfg.epochs + 1):
        if cfg.verbose:
            print(f"\n[reflact] epoch {epoch}/{cfg.epochs}")

        epoch_edits_applied = 0
        epoch_edits_rejected = 0

        # ── Slow update: longitudinal comparison (epoch 2+) ────────────
        if cfg.use_slow_update and epoch >= 2 and all_rollouts:
            if cfg.verbose:
                print(f"  [slow_update] epoch {epoch}: comparing epoch {epoch-1} vs {epoch}")

            # Re-rollout with both skills on the SAME cases
            prev_results = rollout_fn(prev_epoch_skill)
            curr_results = rollout_fn(current_skill)

            from .comparison import build_comparison_pairs, run_slow_update, extract_slow_update_content, replace_slow_update_content

            pairs = build_comparison_pairs(prev_results, curr_results)
            if pairs:
                n_reg = sum(1 for p in pairs if p["category"] == "regressed")
                n_imp = sum(1 for p in pairs if p["category"] == "improved")
                n_per = sum(1 for p in pairs if p["category"] == "persistent_fail")
                n_sta = sum(1 for p in pairs if p["category"] == "stable_success")
                if cfg.verbose:
                    print(f"    comparison: regressed={n_reg} improved={n_imp} "
                          f"persistent_fail={n_per} stable={n_sta}")

                existing = extract_slow_update_content(current_skill)
                slow_result = run_slow_update(
                    current_skill, prev_results, curr_results,
                    optimizer=optimizer, existing_guidance=existing,
                    max_tokens=cfg.optimizer_max_tokens,
                )

                if slow_result and slow_result.get("slow_update_content"):
                    new_guidance = slow_result["slow_update_content"]
                    current_skill = replace_slow_update_content(current_skill, new_guidance)
                    best_skill = replace_slow_update_content(best_skill, new_guidance)
                    slow_updates_applied += 1
                    if cfg.verbose:
                        action = slow_result.get("action", "accept")
                        print(f"    slow_update: {action} ({len(new_guidance)} chars guidance)")

        # ── Meta-skill (start of epoch, except first) ──────────────────
        if cfg.use_meta_skill and epoch > 1 and epoch % cfg.meta_skill_interval == 0 and history:
            try:
                from .meta_skill import run_meta_skill as _run_ms
                ms_result = _run_ms(
                    skill_content=current_skill,
                    history=history,
                    current_meta_skill=active_meta_skill,
                    optimizer=optimizer,
                    max_tokens=cfg.optimizer_max_tokens,
                    epoch=epoch,
                )
                if ms_result.get("meta_skill"):
                    active_meta_skill = ms_result["meta_skill"]
                    if cfg.verbose:
                        print(f"  [meta_skill] updated ({len(active_meta_skill)} chars)")
            except Exception:
                pass

        # ── Context formatting ─────────────────────────────────────────
        step_buffer_context = _format_step_buffer(step_buffer)
        from .meta_skill import format_meta_skill_context as _fmt_ms
        meta_skill_context = _fmt_ms(active_meta_skill)

        # ── Step loop ──────────────────────────────────────────────────
        for step_in_epoch in range(cfg.accumulation):
            global_step += 1
            t_step = time.time()

            if cfg.verbose:
                print(f"  step {global_step} ", end="")

            # ① ROLLOUT
            expanded = rollout_fn(current_skill)
            rolls = expanded[:cfg.batch_size] if len(expanded) > cfg.batch_size else expanded
            scores = [r.get("score", 0.0) for r in rolls]
            avg_score = sum(scores) / max(len(scores), 1)
            failures = [r for r in rolls if r.get("score", 0.0) < 0.5]
            successes = [r for r in rolls if r.get("score", 0.0) >= 0.5]

            if cfg.verbose:
                print(f"rollout={avg_score:.3f} F={len(failures)} S={len(successes)} ", end="")

            # ② REFLECT
            all_failure_patches: list[dict[str, Any]] = []
            all_success_patches: list[dict[str, Any]] = []

            if failures:
                fr = analyze_failures(current_skill, failures, optimizer=optimizer,
                    max_tokens=cfg.optimizer_max_tokens, minibatch_size=cfg.minibatch_size,
                    step_buffer_context=step_buffer_context, meta_skill_context=meta_skill_context)
                all_failure_patches = fr["patches"]
            if successes:
                sr = analyze_successes(current_skill, successes, optimizer=optimizer,
                    max_tokens=cfg.optimizer_max_tokens, minibatch_size=cfg.minibatch_size,
                    step_buffer_context=step_buffer_context, meta_skill_context=meta_skill_context)
                all_success_patches = sr["patches"]

            n_patches = len(all_failure_patches) + len(all_success_patches)
            if cfg.verbose:
                print(f"patches={n_patches} ", end="")

            if not all_failure_patches and not all_success_patches:
                if cfg.verbose:
                    print("→ skip (no patches)")
                continue

            # ③ AGGREGATE
            try:
                from .aggregate import merge_patches as _hierarchical_merge
                merged_patch = _hierarchical_merge(
                    current_skill, all_failure_patches, all_success_patches,
                    optimizer=optimizer, batch_size=cfg.merge_batch_size,
                    max_tokens=cfg.optimizer_max_tokens, workers=cfg.aggregate_workers,
                    verbose=cfg.verbose)
            except Exception:
                merged_patch = {"edits": merge_edits(all_failure_patches + all_success_patches, max_edits=cfg.edit_budget * 2)}

            edits = merged_patch.get("edits", [])
            n_edits_merged = len(edits)
            if cfg.verbose:
                print(f"merged={n_edits_merged} ", end="")
            if not edits:
                if cfg.verbose:
                    print("→ skip")
                continue

            # ④ SELECT
            if cfg.update_mode in ("rewrite_from_suggestions",):
                edit_budget = None; ranked_edits = edits
            elif cfg.lr_scheduler == "autonomous":
                from ._autonomous_lr import decide_autonomous_learning_rate as _alr
                lr_d = _alr(skill_content=current_skill, edits=edits, rollout_score=avg_score,
                             rollout_n=len(rolls), optimizer=optimizer, max_tokens=cfg.optimizer_max_tokens)
                edit_budget = lr_d.get("learning_rate", cfg.edit_budget)
                ranked_edits = _do_rank(current_skill, edits, edit_budget, optimizer, cfg)
            else:
                edit_budget = scheduler.step()
                ranked_edits = _do_rank(current_skill, edits, edit_budget, optimizer, cfg)

            n_selected = len(ranked_edits)
            if cfg.verbose:
                print(f"sel={n_selected}/{n_edits_merged} budget={edit_budget} ", end="")
            if not ranked_edits:
                if cfg.verbose:
                    print("→ skip")
                continue

            # ⑤ UPDATE
            if cfg.update_mode == "rewrite_from_suggestions":
                from ._rewrite_skill import rewrite_skill_from_suggestions as _rws
                rewrite_result = _rws(current_skill, {"edits": ranked_edits}, optimizer=optimizer, max_tokens=cfg.rewrite_max_tokens)
                if rewrite_result and rewrite_result.get("new_skill"):
                    candidate_skill = rewrite_result["new_skill"]; n_applied = len(ranked_edits); n_skipped = 0
                else:
                    candidate_skill = current_skill; n_applied = 0; n_skipped = len(ranked_edits)
            else:
                candidate_skill, reports = apply_patch(current_skill, ranked_edits)
                n_applied = sum(1 for r in reports if r.get("status", "").startswith("applied"))
                n_skipped = len(reports) - n_applied

            if cfg.verbose:
                print(f"applied={n_applied} ", end="")
            if n_applied == 0:
                if cfg.verbose:
                    print("→ skip")
                continue

            # ⑥ EVALUATE
            candidate_rollouts = _eval(candidate_skill)
            candidate_score = _mean_score(candidate_rollouts)
            if cfg.verbose:
                print(f"cand={candidate_score:.3f} ", end="")

            if candidate_score > best_score + cfg.gate_threshold:
                best_skill, best_score, best_step = candidate_skill, candidate_score, global_step
                current_skill, current_score = candidate_skill, candidate_score
                action = "accept_new_best"
                epoch_edits_applied += n_applied; total_edits_applied += n_applied
                consecutive_rejections = 0
            elif candidate_score >= current_score - cfg.gate_threshold:
                current_skill, current_score = candidate_skill, candidate_score
                action = "accept_no_degrade"
                epoch_edits_applied += n_applied; total_edits_applied += n_applied
                consecutive_rejections = 0
            else:
                action = "reject"
                total_edits_rejected += n_applied; epoch_edits_rejected += n_applied
                consecutive_rejections += 1
                step_buffer.append({"step": global_step, "edits": [describe_edit(e) for e in ranked_edits],
                    "candidate_score": candidate_score, "current_score": current_score})
                if len(step_buffer) > 20: step_buffer = step_buffer[-20:]

            if cfg.verbose:
                print(f"→ {action}")

            history.append({"step": global_step, "epoch": epoch, "action": action,
                "rollout_score": round(avg_score, 4), "candidate_score": round(candidate_score, 4),
                "current_score": round(current_score, 4), "best_score": round(best_score, 4),
                "n_patches": n_patches, "n_merged": n_edits_merged, "n_selected": n_selected,
                "n_applied": n_applied, "n_skipped": n_skipped,
                "n_failures": len(failures), "n_successes": len(successes),
                "edit_budget": edit_budget, "wall_time_s": round(time.time() - t_step, 1)})

            if consecutive_rejections >= cfg.max_rejected_steps:
                if cfg.verbose:
                    print(f"  early stop: {consecutive_rejections} consecutive rejections")
                break

        # ── Epoch summary ──────────────────────────────────────────────
        if cfg.verbose:
            print(f"  epoch {epoch} done: applied={epoch_edits_applied} "
                  f"rejected={epoch_edits_rejected} best_score={best_score:.4f}")

        (out_dir / f"best_skill_epoch_{epoch}.md").write_text(best_skill, encoding="utf-8")
        (out_dir / f"current_skill_epoch_{epoch}.md").write_text(current_skill, encoding="utf-8")
        (out_dir / f"checkpoint_epoch_{epoch}.json").write_text(json.dumps({
            "epoch": epoch, "best_score": best_score, "best_step": best_step,
            "current_score": current_score, "skill_len": len(best_skill),
            "meta_skill_len": len(active_meta_skill),
        }, indent=2), encoding="utf-8")

        if active_meta_skill:
            (out_dir / "meta_skill.md").write_text(active_meta_skill, encoding="utf-8")

        # ── Save prev_epoch_skill for next slow update ─────────────────
        prev_epoch_skill = current_skill

        if consecutive_rejections >= cfg.max_rejected_steps:
            break

    # ── Final persist ──────────────────────────────────────────────────
    (out_dir / "best_skill.md").write_text(best_skill, encoding="utf-8")
    (out_dir / "current_skill.md").write_text(current_skill, encoding="utf-8")
    with (out_dir / "train_history.jsonl").open("w", encoding="utf-8") as f:
        for rec in history:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return TrainResult(
        initial_skill=initial_skill, best_skill=best_skill, current_skill=current_skill,
        best_score=best_score, best_step=best_step,
        epochs_completed=epoch if consecutive_rejections < cfg.max_rejected_steps else epoch - 1,
        steps_completed=global_step, total_edits_applied=total_edits_applied,
        total_edits_rejected=total_edits_rejected,
        elapsed_s=round(time.time() - t_start, 1),
        meta_skill=active_meta_skill, slow_updates_applied=slow_updates_applied,
        history=history,
    )


# ── helpers ────────────────────────────────────────────────────────────


def _do_rank(skill, edits, budget, optimizer, cfg):
    if len(edits) <= budget:
        return edits
    from .select import rank_and_select
    return rank_and_select(skill, edits, budget, optimizer=optimizer, optimizer_max_tokens=cfg.optimizer_max_tokens)


def _mean_score(rollouts: list[dict[str, Any]]) -> float:
    if not rollouts:
        return 0.0
    return sum(r.get("score", 0.0) for r in rollouts) / len(rollouts)


def _format_step_buffer(buffer: list[dict[str, Any]]) -> str:
    if not buffer:
        return ""
    lines = ["The following edits were REJECTED in previous steps of this epoch "
             "because they degraded validation score. Do NOT propose similar edits."]
    for entry in buffer[-10:]:
        step = entry.get("step", "?")
        cand = entry.get("candidate_score", 0)
        curr = entry.get("current_score", 0)
        edits = entry.get("edits", [])
        lines.append(f"  Step {step} (candidate={cand:.3f} < current={curr:.3f}):")
        for e in edits[:3]:
            lines.append(f"    REJECTED: {e}")
    return "\n".join(lines)


def load_skill_from_agents_md(repo_root: Path) -> str:
    agents_path = repo_root / "AGENTS.md"
    if agents_path.exists():
        return agents_path.read_text(encoding="utf-8")
    template_path = Path(__file__).parent.parent / "templates" / "AGENTS.md"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "# Agent Skill Document\n\nNo rules defined.\n"
