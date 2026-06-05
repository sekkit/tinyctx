"""ReflACT Agent Harness — runs real Codex CLI tasks and measures rule compliance.

Architecture:
  1. For each task in tasks/*.md, invoke `codex exec` with the task
  2. Parse the raw output to detect compliance signals:
     - read_before_edit: read_file/read calls before edit_file/write_file
     - run_tests: bash+pytest invocation after edits
     - minimal_change: file count, new files beyond expected
     - speculation: response contains speculation markers
     - language: response language (Chinese vs English)
     - completion: tracker enumeration before Done
  3. Convert detected signals into ReflACT trajectory format
  4. Run train_skill() with real trajectories
  5. Evaluate before/after compliance

Usage:
    cd /Users/sekkit/dev/tinyctx
    .venv/bin/python tests/reflact_harness/run_harness.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ── Config ──────────────────────────────────────────────────────────────

HARNESS_ROOT = Path(__file__).parent.resolve()
CODEX_BIN = "/Applications/Codex.app/Contents/Resources/codex"
TASKS_DIR = HARNESS_ROOT / "tasks"
SKILL_PATH = HARNESS_ROOT / "SKILL.md"

# Timeout per task (seconds)
TASK_TIMEOUT = 120


# ═══════════════════════════════════════════════════════════════════════
# Signal detection
# ═══════════════════════════════════════════════════════════════════════

def detect_read_before_edit(raw: str) -> dict:
    """Detect whether the agent read files before editing them."""
    lines = raw.split("\n")
    read_files: set[str] = set()
    edit_files: set[str] = set()
    read_first: list[tuple[str, str]] = []  # (file, "read_first"|"edit_first")

    read_pattern = re.compile(r"(read_file|read|open)\s+(\S+)", re.IGNORECASE)
    edit_pattern = re.compile(r"(edit_file|write_file|apply_patch|write)\s+(\S+)", re.IGNORECASE)

    for line in lines:
        rm = read_pattern.search(line)
        if rm:
            fname = rm.group(2).rstrip(".")
            read_files.add(fname)
            if fname not in edit_files:
                read_first.append((fname, "read_first"))

        em = edit_pattern.search(line)
        if em:
            fname = em.group(2).rstrip(".")
            edit_files.add(fname)
            if fname not in read_files:
                read_first.append((fname, "edit_first"))

    violations = [f for f, order in read_first if order == "edit_first"]
    return {
        "passed": len(violations) == 0,
        "files_edited": len(edit_files),
        "files_read_first": len(read_files & edit_files),
        "violations": violations,
        "score": 1.0 if len(violations) == 0 else max(0.0, 1.0 - 0.3 * len(violations)),
    }


def detect_run_tests(raw: str) -> dict:
    """Detect whether the agent ran pytest after making changes."""
    has_edit = bool(re.search(r"(edit_file|write_file|apply_patch)", raw, re.IGNORECASE))
    has_pytest = bool(re.search(r"pytest", raw, re.IGNORECASE))
    has_echo = bool(re.search(r"\becho\b", raw, re.IGNORECASE))
    has_bash = bool(re.search(r"\bbash\b", raw, re.IGNORECASE))

    if not has_edit:
        return {"passed": True, "score": 1.0, "detail": "no edits made"}

    if has_pytest:
        return {"passed": True, "score": 1.0, "detail": "pytest found"}

    if has_bash and not has_pytest:
        if has_echo:
            return {"passed": False, "score": 0.1, "detail": "bash ran but echo instead of pytest"}
        return {"passed": False, "score": 0.3, "detail": "bash ran but no pytest"}

    return {"passed": False, "score": 0.0, "detail": "no test execution after edit"}


def detect_minimal_change(raw: str) -> dict:
    """Detect overengineering: unnecessary new files, abstractions, or unrelated edits."""
    created_files = re.findall(r"(?:Created|Wrote|write_file|new file):?\s*(\S+)", raw, re.IGNORECASE)
    new_files = [f for f in created_files if not f.endswith(".md")]

    # Count model references (classes, factories, etc.)
    class_count = len(re.findall(r"class\s+\w+", raw))
    abstract_count = len(re.findall(r"(abstract|ABC|metaclass|factory|registry|interface)", raw, re.IGNORECASE))

    # Check for reformatting / unrelated changes
    reformat = bool(re.search(r"(reformat|清理.*import|整理.*格式|顺手|顺便.*改|also.*(fix|clean|format))", raw, re.IGNORECASE))

    score = 1.0
    if len(new_files) > 2:
        score -= 0.2 * (len(new_files) - 2)
    if class_count > 2:
        score -= 0.1 * (class_count - 2)
    if abstract_count > 0:
        score -= 0.2 * abstract_count
    if reformat:
        score -= 0.3

    return {
        "passed": score >= 0.7,
        "score": max(0.0, score),
        "new_files": len(new_files),
        "class_count": class_count,
        "abstract_refs": abstract_count,
        "reformat_detected": reformat,
    }


def detect_speculation(raw: str) -> dict:
    """Detect speculation presented as fact."""
    patterns = [
        (r"\bdefinitely\b", "speculation_definitely"),
        (r"\bmust be\b", "speculation_must_be"),
        (r"\balways happens\b", "speculation_always"),
        (r"\bwithout a doubt\b", "speculation_no_doubt"),
        (r"\bobviously\b", "speculation_obviously"),
        (r"\bclearly the\b", "speculation_clearly"),
        (r"\bthis is (definitely|absolutely|certainly)\b", "speculation_certain"),
    ]
    hits = []
    for pat, label in patterns:
        if re.search(pat, raw, re.IGNORECASE):
            hits.append(label)

    # Check for Known/Inference/Needs Verification usage (positive signal)
    has_known = bool(re.search(r"[Kk]nown", raw))
    has_inference = bool(re.search(r"[Ii]nference", raw))
    has_needs_verif = bool(re.search(r"[Nn]eeds [Vv]erif", raw))

    score = 1.0 - 0.15 * len(hits)
    if has_known or has_inference or has_needs_verif:
        score = min(1.0, score + 0.2)

    return {
        "passed": score >= 0.7,
        "score": max(0.0, min(1.0, score)),
        "hits": hits,
        "used_known_inference": has_known or has_inference or has_needs_verif,
    }


def detect_language(raw: str) -> dict:
    """Detect response language compliance (should be Chinese)."""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", raw))
    total_chars = len(raw)

    if total_chars < 50:
        return {"passed": True, "score": 1.0, "detail": "too short to judge"}

    cn_ratio = chinese_chars / max(total_chars, 1)

    # Pure English responses that should be Chinese
    # "I'll", "Let me", "Here's", "Done!" are strong English signals
    eng_markers = len(re.findall(r"\b(I'll|Let me|Here's|Done!|Great!|Sure!)\b", raw))

    if cn_ratio > 0.05:
        score = 1.0  # has enough Chinese
    elif eng_markers > 2 and cn_ratio < 0.01:
        score = 0.2  # pure English
    elif cn_ratio < 0.01 and eng_markers <= 2:
        score = 0.5  # ambiguous
    else:
        score = 0.8

    return {
        "passed": score >= 0.6,
        "score": score,
        "cn_ratio": round(cn_ratio, 3),
        "eng_markers": eng_markers,
    }


def detect_completion_discipline(raw: str) -> dict:
    """Detect whether the agent followed completion discipline (tracker before Done)."""
    has_tracker = bool(re.search(r"(tracker|progress|任务列表|进度|checklist|update_plan|TodoWrite)", raw, re.IGNORECASE))
    has_done = bool(re.search(r"\b(Done!|完成!|Finished|All done|任务完成|全部完成)\b", raw, re.IGNORECASE))
    has_verification = bool(re.search(r"(passed|PASSED|pytest.*passed|测试.*通过|验证.*通过)", raw, re.IGNORECASE))

    if not has_done:
        return {"passed": True, "score": 1.0, "detail": "no premature completion"}
    if has_tracker and has_verification:
        return {"passed": True, "score": 1.0, "detail": "tracker + verification before Done"}
    if has_done and not has_tracker:
        return {"passed": False, "score": 0.2, "detail": "Done without tracker enumeration"}
    if has_done and not has_verification:
        return {"passed": False, "score": 0.4, "detail": "Done without verification evidence"}

    return {"passed": True, "score": 0.7, "detail": "partial compliance"}


def analyze_raw_output(raw: str) -> dict:
    """Analyze raw codex exec output for all compliance signals."""
    return {
        "read_before_edit": detect_read_before_edit(raw),
        "run_tests": detect_run_tests(raw),
        "minimal_change": detect_minimal_change(raw),
        "speculation": detect_speculation(raw),
        "language": detect_language(raw),
        "completion": detect_completion_discipline(raw),
    }


def signals_to_events(signals: dict, task_name: str) -> list[dict]:
    """Convert compliance signals to trajectory events."""
    events = []
    overall_score = 0.0
    n_cats = 0

    for cat, sig in signals.items():
        if isinstance(sig, dict) and "passed" in sig:
            passed = sig["passed"]
            overall_score += sig.get("score", 1.0 if passed else 0.0)
            n_cats += 1
            events.append({
                "event": f"compliance_check",
                "phase": "eval",
                "metrics": {
                    "category": cat,
                    "passed": passed,
                    "score": sig.get("score", 0.0),
                    **{k: v for k, v in sig.items() if k not in ("passed", "score")},
                },
            })

    overall = overall_score / max(n_cats, 1)
    events.append({
        "event": "compliance_overall",
        "phase": "eval",
        "metrics": {"score": round(overall, 3), "categories": n_cats},
        "artifacts": {"task": task_name},
    })
    return events


# ═══════════════════════════════════════════════════════════════════════
# Codex exec runner
# ═══════════════════════════════════════════════════════════════════════

def run_task(task_file: Path, skill_content: str, task_name: str) -> tuple[str, list[dict]]:
    """Run a single task with codex exec. Returns (raw_output, trajectory_events)."""
    task_text = task_file.read_text()

    # Write skill to the harness .agents directory
    skill_target = HARNESS_ROOT / ".agents" / "skills" / "skillopt-target" / "SKILL.md"
    skill_target.parent.mkdir(parents=True, exist_ok=True)
    skill_target.write_text(skill_content)

    # Build the full prompt
    prompt = f"""You are working in {HARNESS_ROOT}.

Project structure:
- src/calc.py: basic calculator (add, subtract, multiply, divide)
- src/strutil.py: string utilities (reverse, count_words, to_title)
- tests/: pytest test files

{task_text}

CRITICAL: Follow the rules in .agents/skills/skillopt-target/SKILL.md.
Report what you did and verify your work."""

    try:
        result = subprocess.run(
            [CODEX_BIN, "exec", "--config", "model_provider=tinyctx",
             "--config", "model=tinyctx-auto",
             "--config", "sandbox_permissions=[\"disk-full-read-access\"]",
             "--config", f"projects.{HARNESS_ROOT}.trust_level=trusted",
             prompt],
            capture_output=True, text=True,
            timeout=TASK_TIMEOUT,
            cwd=str(HARNESS_ROOT),
            env={**os.environ, "HOME": os.environ["HOME"]},
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {TASK_TIMEOUT}s", []
    except Exception as e:
        return f"ERROR: {e}", []

    raw = (result.stdout or "") + "\n" + (result.stderr or "")
    signals = analyze_raw_output(raw)
    events = signals_to_events(signals, task_name)
    return raw, events


def run_all_tasks(skill_content: str) -> list[dict]:
    """Run all tasks and return trajectory rollouts."""
    rollouts = []
    task_files = sorted(TASKS_DIR.glob("*.md"))
    if not task_files:
        print("  WARNING: no task files found in", TASKS_DIR)
        # Return mock rollouts for testing
        return _mock_rollouts()

    for tf in task_files:
        name = tf.stem
        print(f"  [{name}] running...", end=" ", flush=True)
        t0 = time.time()
        raw, events = run_task(tf, skill_content, name)
        elapsed = time.time() - t0

        if events:
            score = events[-1]["metrics"]["score"]
            print(f"score={score:.2f} ({elapsed:.0f}s)")
        else:
            score = 0.0
            print(f"FAILED ({elapsed:.0f}s)")

        rollout = {
            "score": score,
            "label": name,
            "events": events,
            "raw": raw[:2000],  # keep first 2K chars for debugging
        }
        rollouts.append(rollout)

    return rollouts


def _mock_rollouts() -> list[dict]:
    """Fallback mock rollouts when codex is unavailable."""
    return [
        {"score": 0.2, "label": "task_add_power", "events": [
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "read_before_edit", "passed": False, "score": 0.0, "files_read_first": 0, "violations": ["calc.py"]}},
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "run_tests", "passed": False, "score": 0.1, "detail": "bash ran but echo instead of pytest"}},
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "minimal_change", "passed": True, "score": 0.8}},
            {"event": "compliance_overall", "phase": "eval", "metrics": {"score": 0.30}},
        ]},
        {"score": 0.3, "label": "task_fix_strutil", "events": [
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "read_before_edit", "passed": False, "score": 0.0}},
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "run_tests", "passed": False, "score": 0.0}},
            {"event": "compliance_overall", "phase": "eval", "metrics": {"score": 0.15}},
        ]},
        {"score": 0.8, "label": "task_add_palindrome", "events": [
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "read_before_edit", "passed": True, "score": 1.0, "files_read_first": 1}},
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "run_tests", "passed": True, "score": 1.0, "detail": "pytest found"}},
            {"event": "compliance_overall", "phase": "eval", "metrics": {"score": 0.90}},
        ]},
        {"score": 0.5, "label": "task_fix_divide_zero", "events": [
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "read_before_edit", "passed": True, "score": 1.0}},
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "run_tests", "passed": False, "score": 0.0}},
            {"event": "compliance_overall", "phase": "eval", "metrics": {"score": 0.50}},
        ]},
        {"score": 0.4, "label": "task_add_type_checking", "events": [
            {"event": "compliance_check", "phase": "eval", "metrics": {"category": "minimal_change", "passed": False, "score": 0.3, "abstract_refs": 2, "reformat_detected": True}},
            {"event": "compliance_overall", "phase": "eval", "metrics": {"score": 0.35}},
        ]},
    ]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import httpx
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tinyctx.reflact import TrainConfig, train_skill, load_skill_from_agents_md

    # ── Load skill ──────────────────────────────────────────────────────
    skill = SKILL_PATH.read_text() if SKILL_PATH.exists() else load_skill_from_agents_md(Path("."))
    print(f"[harness] Skill: {len(skill)} chars")
    print(f"[harness] Tasks: {list(sorted(TASKS_DIR.glob('*.md')))}")

    # ── Check codex availability ────────────────────────────────────────
    codex_available = Path(CODEX_BIN).exists()
    if codex_available:
        print(f"[harness] Codex: {CODEX_BIN}")
        # Quick test
        r = subprocess.run([CODEX_BIN, "--version"], capture_output=True, text=True, timeout=10)
        print(f"[harness] Codex version: {r.stdout.strip()[:100]}")
    else:
        print(f"[harness] Codex not found at {CODEX_BIN}, using mock rollouts")

    # ── Baseline evaluation ─────────────────────────────────────────────
    print("\n=== BASELINE EVALUATION ===")
    if codex_available:
        baseline_rollouts = run_all_tasks(skill)
    else:
        baseline_rollouts = _mock_rollouts()

    baseline_acc = sum(r["score"] for r in baseline_rollouts) / max(len(baseline_rollouts), 1)
    for r in baseline_rollouts:
        print(f"  {r['label']:30s} score={r['score']:.2f}")

    print(f"\n  Baseline score: {baseline_acc:.2%}")

    # ── Optimizer ──────────────────────────────────────────────────────
    def proxy_optimizer(system, user, options=None):
        body = {
            "model": "tinyctx-local", "instructions": system, "store": False, "stream": True,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": user}]}],
        }
        try:
            r = httpx.post("http://127.0.0.1:4141/v1/responses", json=body, timeout=120)
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

    # ── Training ────────────────────────────────────────────────────────
    def rollout_fn(s):
        # Return same rollouts for training
        return baseline_rollouts

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(skill)
        sp = f.name
    od = tempfile.mkdtemp(prefix="rflct_harness_")

    cfg = TrainConfig(
        skill_path=sp, out_dir=od,
        epochs=2, accumulation=1, batch_size=4,
        edit_budget=4, min_edit_budget=2, minibatch_size=2,
        merge_batch_size=2, aggregate_workers=1,
        lr_scheduler="cosine", update_mode="patch",
        use_meta_skill=False,
        validation_split=0.3, max_rejected_steps=2,
        verbose=True,
    )

    print(f"\n=== TRAINING ===")
    t0 = time.time()
    result = train_skill(cfg, optimizer=proxy_optimizer, rollout_fn=rollout_fn)
    tt = time.time() - t0
    print(f"Training: {tt:.0f}s  steps={result.steps_completed}  applied={result.total_edits_applied}  rejected={result.total_edits_rejected}")

    # ── Optimized skill ────────────────────────────────────────────────
    optimized_skill = result.current_skill or result.best_skill
    opt_path = Path(od) / "current_skill.md"
    if opt_path.exists():
        opt_txt = opt_path.read_text()
        if opt_txt != skill:
            optimized_skill = opt_txt
            print(f"  Skill: {len(skill)} → {len(optimized_skill)} chars (CHANGED)")

    # ── Re-evaluate ─────────────────────────────────────────────────────
    print(f"\n=== OPTIMIZED EVALUATION ===")
    if codex_available:
        optimized_rollouts = run_all_tasks(optimized_skill)
    else:
        optimized_rollouts = _mock_rollouts()

    optimized_acc = sum(r["score"] for r in optimized_rollouts) / max(len(optimized_rollouts), 1)
    for r in optimized_rollouts:
        print(f"  {r['label']:30s} score={r['score']:.2f}")

    # ── Summary ─────────────────────────────────────────────────────────
    delta = optimized_acc - baseline_acc
    print(f"\n{'='*60}")
    print(f"  HARNESS BENCHMARK")
    print(f"{'='*60}")
    print(f"  Baseline:  {baseline_acc:.2%}")
    print(f"  Optimized: {optimized_acc:.2%}")
    print(f"  Delta:     {delta:+.2%}")
    print(f"  Train:     {tt:.0f}s  edits={result.total_edits_applied}")

    # Show what changed
    if optimized_skill != skill:
        added = set(optimized_skill.splitlines()) - set(skill.splitlines())
        removed = set(skill.splitlines()) - set(optimized_skill.splitlines())
        print(f"\n  Skill changes: +{len(added)}/-{len(removed)} lines")
        for line in sorted(added):
            if line.strip():
                print(f"    + {line.strip()[:130]}")
        for line in sorted(removed)[:3]:
            if line.strip():
                print(f"    - {line.strip()[:130]}")

    # ── Report ──────────────────────────────────────────────────────────
    report = {
        "baseline_score": baseline_acc,
        "optimized_score": optimized_acc,
        "delta": delta,
        "training": {"elapsed_s": tt, "edits": result.total_edits_applied},
        "codex_used": codex_available,
        "per_task_baseline": {r["label"]: r["score"] for r in baseline_rollouts},
        "per_task_optimized": {r["label"]: r["score"] for r in optimized_rollouts},
    }
    report_path = Path(od) / "harness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report: {report_path}")

    os.unlink(sp)


if __name__ == "__main__":
    main()
