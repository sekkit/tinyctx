# tinyctx

**Local-first context-routing proxy for OpenAI Codex CLI.** 95%+ of turns run on cheap models (DeepSeek-V4-Flash). Frontier (GPT-5.5) only fires when truly needed. Zero manual config — all integrations auto-install and auto-wire.

```bash
./scripts/install.sh      # one-shot: venv + all MCP servers + codex config
./scripts/start.sh        # starts proxy on 127.0.0.1:4141
codex --profile tinyctx   # uses the proxy as model_provider
```

## Highlights

- **~95% local hit rate in production.** DeepSeek-V4-Flash handles routine turns at ~1/100th frontier cost. Compaction summaries, repeat reads, and tool results never touch GPT-5.5.
- **7 OSS integrations, all auto.** gitnexus (knowledge graph), graphify (code skill), serena (LSP ops), caveman-shrink (compression), context-mode (sandbox), mem0 (memory), LLMLingua-2 — one `install.sh` wires everything.
- **Inline compression pipeline.** Tool descriptions shortened 30-50% via regex rules. Repeat reads collapsed to unified diffs. Cache-aware mutation gates preserve prompt-cache hits.
- **Advisor Strategy built-in.** Classifier detects hard decisions (p≥0.7) → auto-escalates to GPT-5.5 as a sub-agent. executor handles 99% of turns; frontier only sees 1-3 hard calls per task.
- **Live dashboard.** Per-project token stats, MCP call counts, frontier health, route mix, latency percentiles — all at `http://127.0.0.1:4141/dashboard`.
- **DeepSeek native.** `reasoning_content` streaming, tool schema pass-through, chat→responses SSE bridge, codex 0.128+ wire compat — all production-verified at 700+ turns/day.

## Architecture

```
codex CLI
  │  Responses API
  ▼
tinyctx-proxy (FastAPI, :4141)
  │
  ├─ route 95% → DeepSeek-V4-Flash (api.deepseek.com/v1)
  │
  └─ escalate 5% → GPT-5.5 (chatgpt.com/backend-api/codex)
       ↑ triggered by: image detected, error streak ≥2,
          self-classify p≥0.7, or guard pipeline force

In parallel, codex talks directly to auto-installed MCP servers:
  gitnexus · graphify · serena · caveman-shrink · context-mode · advisor
```

## Quick start

```bash
cd ~/dev/tinyctx
./scripts/install.sh                    # idempotent, safe to re-run
cp examples/secrets.env.example ~/.tinyctx/secrets.env
# edit secrets.env → paste your DeepSeek API key

./scripts/start.sh                      # proxy on :4141, dashboard auto-opens
codex --profile tinyctx                 # routes through proxy
codex --profile tinyctx-goal            # for long-running goal mode
```

Dashboard at `http://127.0.0.1:4141/dashboard` — shows token stats, route mix, MCP call counts, project tabs.

## Key numbers

| Metric | Value |
|--------|-------|
| Local hit rate (production) | ~95% |
| DeepSeek-V4-Flash latency | ~5-7 s/turn |
| GPT-5.5 latency | ~20-30 s/turn |
| Compaction redirect saving | 100% (never hits frontier) |
| Tests | 1005 passing, 31 test files |
| Code | ~13K LOC in 71 modules |

## Config

`~/.tinyctx/config.toml` — all defaults work out of the box. Common tweaks:

```toml
[routing]
force_route = "auto"                          # "local" | "frontier" | "auto"
escalate_on_error_streak = 2                  # consecutive errors → frontier
self_classify_escalates_to_frontier = true    # AI decides when to escalate

[local]
base_url = "https://api.deepseek.com/v1"      # DeepSeek API (default)
model = "deepseek-v4-flash"
context_window = 131072

[frontier]
base_url = "https://chatgpt.com/backend-api/codex"
model = "gpt-5.5"
proxy = "direct"                               # or "http://127.0.0.1:10809" for tunnel
```

Secrets (`TINYCTX_LOCAL_API_KEY`, etc.) live in `~/.tinyctx/secrets.env` (chmod 600), never in config.toml.

## CLIs

```bash
tinyctx install          # auto-install all missing components
tinyctx status           # check all component states
tinyctx-stats            # routing summary from logs (--quality for S-F grade)
tinyctx-trace --watch    # live request trace
tinyctx-dreamer run      # periodic maintenance: scout + keypin + GC
```

## Docs

- [docs/install.md](docs/install.md) — full install guide, every backend option
- [docs/features.md](docs/features.md) — complete feature inventory
- [docs/test.md](docs/test.md) — test plan and benchmark suite
- [docs/advisor.md](docs/advisor.md) — Advisor Strategy deep-dive

## Status

v0.8.0 — production use at 700+ turns/day, 95%+ local hit rate, sub-1% upstream errors. Works with Codex CLI 0.125.0 and Codex.app 0.128.0+.

MIT license. ~13K LOC original code across 71 modules. 7 upstream OSS integrations auto-wired.
