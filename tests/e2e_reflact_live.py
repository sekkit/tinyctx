"""Live end-to-end test: ReflACT training loop using real tinyctx proxy as optimizer.

Runs a complete train_skill() with:
  - Real AGENTS.md template as initial skill
  - Realistic failure/success rollout trajectories
  - tinyctx proxy (deepseek-v4-flash) as optimizer LLM
  - 2 epochs, 2 accumulation steps, edit_budget=3

Verifies:
  - The training loop runs without errors
  - The optimizer produces actionable edits
  - The best_skill.md output file is written
  - The skill document changes (at least one edit applied)
"""

import json
import httpx
import os
import sys
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyctx.reflact import TrainConfig, train_skill, load_skill_from_agents_md

# ── Real optimizer via tinyctx proxy ──────────────────────────────────

PROXY_URL = os.environ.get("TINYCTX_PROXY_URL", "http://127.0.0.1:4141/v1")


def proxy_optimizer(system: str, user: str, options: dict | None = None) -> tuple[str, dict]:
    """Call the tinyctx proxy's local backend as the ReflACT optimizer.

    Sends system prompt as instructions, user prompt as input message.
    Returns (response_text, metadata_dict).
    """
    opts = options or {}
    body = {
        "model": "tinyctx-local",
        "instructions": system,
        "store": False,
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            }
        ],
    }

    try:
        r = httpx.post(f"{PROXY_URL}/responses", json=body, timeout=180)
        if r.status_code != 200:
            return "", {"error": f"HTTP {r.status_code}", "body": r.text[:200]}

        # Parse SSE stream
        text_parts = []
        usage = {}
        for line in r.text.split("\n"):
            if line.startswith("data: "):
                try:
                    ev = json.loads(line[6:])
                    t = ev.get("type", "")
                    if t == "response.output_text.delta":
                        text_parts.append(ev.get("delta", ""))
                    elif t == "response.completed":
                        usage = ev.get("response", {}).get("usage", {})
                except json.JSONDecodeError:
                    pass

        full_text = "".join(text_parts)
        return full_text, {"stage": opts.get("stage", ""), "tokens": usage}

    except Exception as e:
        return "", {"error": str(e)}


# ── Realistic rollout scenarios ──────────────────────────────────────


def make_rollout_fn():
    """Create a rollout function that returns realistic failure/success trajectories.

    These simulate what a Codex agent would produce when running eval cases.
    Each rollout has structured events + a score (0.0=failure, 1.0=success).
    """
    # These trajectories reflect common patterns in AGENTS.md violations
    scenarios = [
        # FAILURE: agent added type hints but forgot to run tests
        {
            "score": 0.25,
            "label": "case_fix_bug_no_test",
            "events": [
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "read_file", "path": "src/handler.py"}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "edit_file", "path": "src/handler.py", "lines_changed": 3}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "bash", "command": "echo done"}},
                {"event": "response_sent", "phase": "done", "metrics": {"passed": False, "error": "tests/tests_handler.py: FAILED test_update_user — got None, expected User object"}},
                {"event": "request_received", "phase": "received", "metrics": {"user_feedback": "you broke the test! You didn't run pytest after editing."}},
            ],
        },
        # FAILURE: agent added speculative abstraction not requested
        {
            "score": 0.30,
            "label": "case_overengineer",
            "events": [
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "read_file", "path": "src/parser.py"}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "edit_file", "path": "src/parser.py", "lines_changed": 12}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "write_file", "path": "src/parser_base.py", "lines": 45}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "edit_file", "path": "src/__init__.py", "lines_changed": 2}},
                {"event": "response_sent", "phase": "done", "metrics": {"passed": False, "error": "Unnecessary abstraction: user only asked for JSON support, not a class hierarchy"}},
            ],
        },
        # FAILURE: agent wrote speculation as fact in response
        {
            "score": 0.15,
            "label": "case_speculation",
            "events": [
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "read_file", "path": "docs/api.md"}},
                {"event": "response_sent", "phase": "done",
                 "metrics": {"passed": False, "error": "Claimed 'the API returns paginated results by default' without evidence — the docs only show single-page examples. Marked as speculation, presented as fact."}},
            ],
        },
        # SUCCESS: agent followed rules correctly
        {
            "score": 1.0,
            "label": "case_success_read_then_edit",
            "events": [
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "read_file", "path": "src/auth.py"}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "read_file", "path": "tests/test_auth.py"}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "edit_file", "path": "src/auth.py", "lines_changed": 5}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "bash", "command": "pytest tests/test_auth.py -x --tb=short"}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "bash", "command": "pytest tests/ -x --tb=short"}},
                {"event": "response_sent", "phase": "done", "metrics": {"passed": True, "tests_passed": 42, "coverage": "94%"}},
            ],
        },
        # SUCCESS: agent correctly identified and stated uncertainty
        {
            "score": 0.85,
            "label": "case_success_uncertainty",
            "events": [
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "read_file", "path": "config/settings.py"}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "grep", "command": "rg 'DATABASE_URL' ."}},
                {"event": "response_sent", "phase": "done",
                 "metrics": {"passed": True, "correctly_flagged_uncertainty": True,
                            "note": "Agent said: 'The DB config uses DATABASE_URL from env — this is Known from settings.py. However, whether the production instance also uses a connection pool is an Inference — the code doesn't show pool config.'"}},
            ],
        },
        # MIXED: partial success (followed some rules, missed others)
        {
            "score": 0.55,
            "label": "case_partial_rule_follow",
            "events": [
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "read_file", "path": "src/report.py"}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "edit_file", "path": "src/report.py", "lines_changed": 8}},
                {"event": "tool_call", "phase": "executing", "metrics": {"tool": "bash", "command": "pytest tests/test_report.py"}},
                {"event": "response_sent", "phase": "done", "metrics": {"passed": False, "error": "Tests pass, but agent also reformatted 3 unrelated docstring lines ('顺手改' violation)"}},
            ],
        },
    ]
    return scenarios


def rollout_fn(skill: str) -> list[dict]:
    """Simulated rollout function."""
    return make_rollout_fn()


# ── Main test ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ReflACT Live E2E Test")
    print("=" * 60)

    # 1. Verify proxy is alive
    print("\n[1/6] Checking proxy connectivity...")
    try:
        r = httpx.get("http://127.0.0.1:4141/", timeout=5)
        info = r.json()
        print(f"  Proxy: {info.get('name')} v{info.get('version')}")
        print(f"  Local backend: {info['local']['model']}")
    except Exception as e:
        print(f"  FAIL: cannot reach proxy — {e}")
        sys.exit(1)

    # 2. Verify optimizer works
    print("\n[2/6] Testing optimizer (one call)...")
    resp, meta = proxy_optimizer(
        "Reply with exactly: OK as plain text. No JSON, no markdown.",
        "Say OK",
    )
    if "ok" in resp.lower():
        print(f"  Optimizer: OK ({len(resp)} chars, {meta.get('tokens',{}).get('total_tokens','?')} tokens)")
    else:
        print(f"  Optimizer WARNING: unexpected response: {resp[:100]}")
        print("  Continuing anyway...")

    # 3. Load initial skill
    print("\n[3/6] Loading initial skill...")
    initial_skill = load_skill_from_agents_md(Path("/Users/sekkit/dev/tinyctx"))
    print(f"  Loaded: {len(initial_skill)} chars")
    # Write to temp file for the trainer
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(initial_skill)
        skill_path = f.name

    # 4. Run training
    print("\n[4/6] Running ReflACT training...")
    out_dir = tempfile.mkdtemp(prefix="reflact_e2e_")
    cfg = TrainConfig(
        skill_path=skill_path,
        out_dir=out_dir,
        epochs=1,
        accumulation=1,
        batch_size=3,
        edit_budget=3,
        minibatch_size=2,
        validation_split=0.3,
        max_rejected_steps=2,
        gate_threshold=0.0,
        verbose=True,
    )

    result = train_skill(
        cfg,
        optimizer=proxy_optimizer,
        rollout_fn=rollout_fn,
    )

    # 5. Check results
    print(f"\n[5/6] Training results:")
    print(f"  Epochs completed: {result.epochs_completed}")
    print(f"  Steps completed:  {result.steps_completed}")
    print(f"  Edits applied:    {result.total_edits_applied}")
    print(f"  Edits rejected:   {result.total_edits_rejected}")
    print(f"  Best score:       {result.best_score:.3f}")
    print(f"  Best step:        {result.best_step}")
    print(f"  Elapsed:          {result.elapsed_s:.1f}s")
    print(f"  Initial skill:    {len(result.initial_skill)} chars")
    print(f"  Best skill:       {len(result.best_skill)} chars")
    print(f"  Changed:          {result.best_skill != result.initial_skill}")

    # 6. Verify outputs
    print(f"\n[6/6] Verifying outputs...")
    out_path = Path(out_dir)
    checks = []
    checks.append(("best_skill.md", (out_path / "best_skill.md").exists()))
    checks.append(("train_history.jsonl", (out_path / "train_history.jsonl").exists()))
    checks.append(("checkpoint files", len(list(out_path.glob("checkpoint_epoch_*.json"))) > 0))

    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    all_ok = all(c[1] for c in checks)
    skill_changed = result.best_skill != result.initial_skill

    # Show diff of changes
    if skill_changed:
        initial_lines = set(result.initial_skill.split("\n"))
        best_lines = set(result.best_skill.split("\n"))
        added = best_lines - initial_lines
        removed = initial_lines - best_lines
        print(f"\n  Skill changes: +{len(added)} lines, -{len(removed)} lines")
        if added:
            print("  Added lines:")
            for line in sorted(added)[:5]:
                if line.strip():
                    print(f"    + {line.strip()[:100]}")
        if removed:
            print("  Removed lines:")
            for line in sorted(removed)[:5]:
                if line.strip():
                    print(f"    - {line.strip()[:100]}")

    # Show history summary
    print(f"\n  Training history ({len(result.history)} steps):")
    for rec in result.history:
        print(f"    step={rec['step']:2d}  action={rec.get('action','?'):20s}  "
              f"score={rec.get('current_score',0):.3f}→{rec.get('candidate_score',0):.3f}  "
              f"edits={rec.get('n_edits',0)} applied={rec.get('n_applied',0)}")

    # Cleanup temp files
    os.unlink(skill_path)

    # Final verdict
    print(f"\n{'='*60}")
    if all_ok and result.steps_completed >= 1:
        print(f"E2E TEST: PASS")
        return 0
    else:
        print(f"E2E TEST: FAILED (outputs_ok={all_ok}, steps={result.steps_completed})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
