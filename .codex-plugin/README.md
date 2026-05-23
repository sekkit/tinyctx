# tinyctx (Codex marketplace plugin)

Install:

```bash
codex plugin marketplace add github:sekkit/tinyctx
codex plugin install tinyctx
```

Then run the proxy and use the bundled profile:

```bash
~/.tinyctx/scripts/start.sh           # in another terminal
codex --profile tinyctx
```

The plugin registers:

- `model_providers.tinyctx` — the local-first router proxy at `127.0.0.1:4141`.
- `profiles.tinyctx` — preset that uses the proxy with sensible context-window defaults.
- `profiles.tinyctx-goal` — goal-mode preset with long context, high execution reasoning, xhigh planning, and `features.goals=true`.
- `skills/goal-forge` — a bundled Goal Forge skill for turning rough coding ideas into `/goal`-ready `SPEC.md` and `GOAL.md` contracts.
- A `SessionStart` hook that warns if the proxy isn't running.

Invoke the bundled skill from Codex as `$tinyctx:goal-forge` when plugin-prefixed skills are shown, or by asking to forge a goal / create a `/goal` contract.
Run long goals with `codex --profile tinyctx-goal` only in trusted project directories; that profile uses `approval_policy="never"` and `sandbox_mode="danger-full-access"` so the run can continue unattended.

Source: https://github.com/sekkit/tinyctx
