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
- A `SessionStart` hook that warns if the proxy isn't running.

Source: https://github.com/sekkit/tinyctx
