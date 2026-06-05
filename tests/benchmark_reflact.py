"""ReflACT Benchmark — keyword-based skill coverage scoring.

Matches skill document content against failure patterns using keyword sets.
Instant (no LLM calls), deterministic, reproducible.

Each category has a set of keywords. A skill "addresses" a pattern if it
contains enough matching keywords for that category.

Then: run ReflACT training, measure before/after keyword coverage.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

PROXY_URL = "http://127.0.0.1:4141/v1"

# ── Keyword definitions per failure category ──────────────────────────

CATEGORY_KEYWORDS = {
    "read_before_edit": {
        "required": ["先读", "read", "读取", "打开", "查看"],
        "bonus": ["先读再改", "read.*before.*edit", "不要没读就改", "inspect.*before"],
    },
    "minimal_change": {
        "required": ["最小改动", "minimal change", "不新增", "不要重构", "局部"],
        "bonus": ["无关", "顺手", "不主动删除", "不做无关", "不新增用户未要求", "默认最小改动"],
    },
    "run_tests": {
        "required": ["pytest", "测试", "test", "验证"],
        "bonus": ["改后.*测试", "run.*test.*after", "每.*改.*测试", "执行.*验证"],
    },
    "speculation": {
        "required": ["推测", "事实", "证据", "不胡编"],
        "bonus": ["Needs Verification", "Inference", "不把推测写成事实", "证据不足"],
    },
    "completion": {
        "required": ["tracker", "tracker", "验证", "完成"],
        "bonus": ["收工", "traker.*enum", "逐条.*enumerate", "不可.*收工", "Done.*之前.*tracker",
                  "completion.*discipline", "不允许未验证", "commit.*hash", "verify before done"],
    },
    "language": {
        "required": ["简体中文", "中文"],
        "bonus": ["默认简体中文", "代码标识符.*保持原样", "命令.*日志.*报错保持原样"],
    },
}

# Count each unaddressed pattern as a gap (baseline has more gaps)
FAILURE_PATTERNS_PER_CATEGORY = {
    "read_before_edit": 4,
    "minimal_change": 3,
    "run_tests": 3,
    "speculation": 4,
    "completion": 3,
    "language": 3,
}


def score_category(skill_text: str, cat: str) -> float:
    """Score 0.0-1.0 for how well a skill addresses a failure category."""
    import re
    kw = CATEGORY_KEYWORDS.get(cat, {"required": [], "bonus": []})
    text_lower = skill_text.lower()

    required_hits = 0
    for pattern in kw["required"]:
        if re.search(pattern, text_lower):
            required_hits += 1

    bonus_hits = 0
    for pattern in kw["bonus"]:
        if re.search(pattern, text_lower):
            bonus_hits += 1

    # Score: each required keyword = 0.25, bonus = 0.1, capped at 1.0
    score = min(1.0, required_hits * 0.25 + bonus_hits * 0.10)
    return score


def evaluate_skill(skill_text: str) -> dict:
    """Score a skill document against all categories."""
    by_cat = {}
    for cat in CATEGORY_KEYWORDS:
        s = score_category(skill_text, cat)
        patterns = FAILURE_PATTERNS_PER_CATEGORY.get(cat, 3)
        addressed = int(s * patterns)
        by_cat[cat] = {
            "score": round(s, 3),
            "patterns_total": patterns,
            "patterns_addressed": addressed,
            "accuracy": round(s, 3),
        }

    total_patterns = sum(v["patterns_total"] for v in by_cat.values())
    total_addressed = sum(v["patterns_addressed"] for v in by_cat.values())
    return {
        "total": {"addressed": total_addressed, "total": total_patterns,
                  "accuracy": round(total_addressed / max(total_patterns, 1), 3)},
        "by_category": {k: dict(v) for k, v in by_cat.items()},
    }


def print_results(results: dict, title: str = ""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    t = results["total"]
    print(f"\n  Coverage: {t['accuracy']:.0%} ({t['addressed']}/{t['total']} patterns)")

    cat_names = {
        "read_before_edit": "先读再改", "minimal_change": "最小改动",
        "run_tests": "改后跑测试", "speculation": "事实边界",
        "completion": "收工纪律", "language": "中文规范",
    }

    print(f"\n  By category:")
    for cat in sorted(results["by_category"]):
        d = results["by_category"][cat]
        name = cat_names.get(cat, cat)
        bar = "█" * int(d["accuracy"] * 20) + "░" * (20 - int(d["accuracy"] * 20))
        print(f"    {name:12s}  {d['accuracy']:.0%}  {bar}  ({d['patterns_addressed']}/{d['patterns_total']})")

    return t["accuracy"]


def main():
    from tinyctx.reflact import TrainConfig, train_skill, load_skill_from_agents_md

    # ── Load skill ─────────────────────────────────────────────────────
    initial_skill = load_skill_from_agents_md(Path(__file__).parent.parent)
    print(f"[bench] Initial skill: {len(initial_skill)} chars")

    # ── Baseline (instant — keyword matching) ──────────────────────────
    baseline = evaluate_skill(initial_skill)
    baseline_acc = print_results(baseline, "BASELINE (AGENTS.md original)")

    # ── Optimizer ──────────────────────────────────────────────────────
    def proxy_optimizer(system, user, options=None):
        body = {
            "model": "tinyctx-local",
            "instructions": system,
            "store": False, "stream": True,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": user}]}],
        }
        try:
            r = httpx.post(f"{PROXY_URL}/responses", json=body, timeout=120)
        except Exception as e:
            return "", {"error": str(e)}
        if r.status_code != 200:
            return "", {"error": f"HTTP {r.status_code}"}
        text = ""
        for line in r.text.split("\n"):
            if line.startswith("data: "):
                try:
                    ev = json.loads(line[6:])
                    if ev.get("type") == "response.output_text.delta":
                        text += ev.get("delta", "")
                except json.JSONDecodeError:
                    pass
        return text, {}

    # Build training rollouts from failure patterns (focused on under-addressed categories)
    def rollout_fn(skill):
        rollouts = []
        for cat in CATEGORY_KEYWORDS:
            for i in range(FAILURE_PATTERNS_PER_CATEGORY.get(cat, 3)):
                rollouts.append({
                    "score": 0.15,
                    "label": f"gap_{cat}_{i}",
                    "events": [
                        {"event": "rule_violation", "phase": "eval",
                         "metrics": {"passed": False, "category": cat,
                                    "detail": f"Agent violated {cat} rule #{i}: skill missing guidance"},
                         "artifacts": {"category": cat}},
                    ],
                })
        # Add some success cases
        for i in range(8):
            rollouts.append({
                "score": 0.9, "label": f"success_{i}",
                "events": [{"event": "success", "phase": "eval", "metrics": {"passed": True}}],
            })
        return rollouts

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(initial_skill)
        skill_path = f.name
    out_dir = tempfile.mkdtemp(prefix="rflct_bench_")

    cfg = TrainConfig(
        skill_path=skill_path, out_dir=out_dir,
        epochs=2, accumulation=1, batch_size=6,
        edit_budget=4, min_edit_budget=2, minibatch_size=3,
        merge_batch_size=3, aggregate_workers=2,
        lr_scheduler="cosine", update_mode="patch",
        use_meta_skill=False,
        validation_split=0.3, max_rejected_steps=2,
        verbose=True,
    )

    print(f"\n[bench] Training: {cfg.epochs} epochs × {cfg.accumulation} steps (budget={cfg.edit_budget}, cosine)")
    t_train = time.time()
    result = train_skill(cfg, optimizer=proxy_optimizer, rollout_fn=rollout_fn)
    train_time = time.time() - t_train
    print(f"[bench] Training: {train_time:.0f}s  steps={result.steps_completed}  "
          f"applied={result.total_edits_applied}  rejected={result.total_edits_rejected}")

    # ── Optimized ──────────────────────────────────────────────────────
    optimized_skill = result.current_skill or result.best_skill
    opt_path = Path(out_dir) / "current_skill.md"
    if opt_path.exists():
        opt_txt = opt_path.read_text()
        if opt_txt != initial_skill:
            optimized_skill = opt_txt

    print(f"  Skill: {len(initial_skill)} → {len(optimized_skill)} chars (changed: {optimized_skill != initial_skill})")

    # ── Evaluate optimized ─────────────────────────────────────────────
    optimized = evaluate_skill(optimized_skill)
    optimized_acc = print_results(optimized, "OPTIMIZED (ReflACT trained)")

    # ── Summary ────────────────────────────────────────────────────────
    delta = optimized_acc - baseline_acc
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:    {baseline_acc:.0%}     ({baseline['total']['addressed']}/{baseline['total']['total']} patterns)")
    print(f"  Optimized:   {optimized_acc:.0%}     ({optimized['total']['addressed']}/{optimized['total']['total']} patterns)")
    print(f"  Delta:       {delta:+.0%}")
    print(f"  Train time:  {train_time:.0f}s")
    print(f"  Edits:       {result.total_edits_applied}")

    print(f"\n  Per-category deltas:")
    cat_names = {"read_before_edit":"先读再改","minimal_change":"最小改动",
                 "run_tests":"改后跑测试","speculation":"事实边界",
                 "completion":"收工纪律","language":"中文规范"}
    for cat in sorted(baseline["by_category"]):
        b = baseline["by_category"][cat]["accuracy"]
        o = optimized["by_category"][cat]["accuracy"]
        d = o - b
        name = cat_names.get(cat, cat)
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        print(f"    {name:12s}  {b:.0%} → {o:.0%}  {arrow} {d:+.0%}  (+{optimized['by_category'][cat]['patterns_addressed'] - baseline['by_category'][cat]['patterns_addressed']} patterns)")

    # ── Skill diff ─────────────────────────────────────────────────────
    if optimized_skill != initial_skill:
        added = set(optimized_skill.splitlines()) - set(initial_skill.splitlines())
        removed = set(initial_skill.splitlines()) - set(optimized_skill.splitlines())
        print(f"\n  New rules added ({len(added)} lines):")
        for line in sorted(added):
            if line.strip():
                print(f"    + {line.strip()[:130]}")
        if removed:
            print(f"\n  Rules removed ({len(removed)} lines):")
            for line in sorted(removed):
                if line.strip():
                    print(f"    - {line.strip()[:130]}")

    # ── Report ─────────────────────────────────────────────────────────
    report = {
        "baseline": {"accuracy": baseline_acc, "by_category": baseline["by_category"]},
        "optimized": {"accuracy": optimized_acc, "by_category": optimized["by_category"]},
        "delta": delta,
        "training": {"elapsed_s": train_time, "steps": result.steps_completed,
                     "edits_applied": result.total_edits_applied},
    }
    report_path = Path(out_dir) / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report: {report_path}")

    os.unlink(skill_path)
    return report


if __name__ == "__main__":
    main()
