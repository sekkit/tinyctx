# Codex Goal Config Checklist

Use this checklist when preparing long-running Codex `/goal` work.

## Required Checks

- Codex CLI version supports `/goal`.
- The project path is trusted.
- Git status is understood before the run starts.
- The working tree does not contain unrelated risky changes.
- The spec has user-approved measurable `done_when` criteria.
- The goal has a scorecard with a metric or checklist, threshold, regression checks, scoring method, and stop condition.
- The goal has a fast feedback loop, or an explicit reason repeated scoring must use a slower cadence.
- Long-running or exploratory goals have working-memory files such as `PLAN.md`, `ATTEMPTS.md`, and `NOTES.md`, or repo-specific equivalents.
- Verification commands are known and can run without interactive prompts.

## Autonomous /goal Settings

This is the native-Codex target config for autonomous, multi-hour `/goal` sessions:

```toml
model = "gpt-5.5"
model_context_window = 1050000
model_auto_compact_token_limit = 997500
model_reasoning_effort = "high"
plan_mode_reasoning_effort = "xhigh"
approval_policy = "never"
sandbox_mode = "danger-full-access"

[features]
goals = true
```

For tinyctx-backed goal runs, prefer a dedicated profile so the default `tinyctx`
profile can stay conservative:

```toml
[profiles.tinyctx-goal]
model_provider = "tinyctx"
model = "tinyctx-auto"
model_context_window = 1050000
model_auto_compact_token_limit = 997500
model_reasoning_effort = "high"
plan_mode_reasoning_effort = "xhigh"
approval_policy = "never"
sandbox_mode = "danger-full-access"
features = { goals = true }
```

Why each setting matters:

- `model = "gpt-5.5"` selects the frontier model for a native run; `model_provider = "tinyctx"` + `model = "tinyctx-auto"` routes through the tinyctx proxy.
- `model_context_window = 1050000` gives the long context budget expected by extended goal runs.
- `model_auto_compact_token_limit = 997500` allows `/goal` sessions to compact context before they hard-stop near the limit.
- `model_reasoning_effort = "high"` is for execution tasks; `medium` is usually too shallow for multi-hour autonomous work.
- `plan_mode_reasoning_effort = "xhigh"` is for the planning pass before execution.
- `approval_policy = "never"` lets the run continue without pausing for approvals.
- `sandbox_mode = "danger-full-access"` gives the run full filesystem access.
- `[features] goals = true` enables goal workflows when the CLI build requires the feature flag.

Only recommend `approval_policy = "never"` and `sandbox_mode = "danger-full-access"` for directories explicitly marked trusted in config. These settings give the model unsupervised write access to the filesystem. Do not use them in a directory you would not let an unsupervised process touch. Prefer applying them only to specific trusted project paths when Codex supports scoped config.

## Safer Middle Ground

For goals that need repo file writes but should not have broad system access, recommend a narrower profile:

```toml
approval_policy = "on-failure"
sandbox_mode = "workspace-write"
```

Use the most permissive settings only when the goal truly needs unattended execution and the project directory is trusted.

## Recommendation Style

Report gaps and risks. Do not silently change config.

If `/goal` is unavailable, still generate `GOAL.md`; tell the user to update Codex before running it as a persisted goal.
