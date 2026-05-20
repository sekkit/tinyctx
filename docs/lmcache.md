# LMCache integration

tinyctx does not embed LMCache. Keep LMCache below the local backend boundary by
running a vLLM or SGLang server with LMCache enabled, then point tinyctx's
`[local]` backend at that OpenAI-compatible endpoint.

## Why this shape

- tinyctx stays a thin routing proxy: no KV-cache lifecycle, eviction, prefill
  orchestration, or LMCache Python dependency.
- vLLM/SGLang own inference and KV reuse.
- tinyctx preserves its existing cache discipline: stable prompt prefixes,
  local compaction routing, and optional history hygiene.

## tinyctx setting

Use `lmcache_passthrough = true` only when your local vLLM/SGLang+LMCache stack
accepts or consumes `prompt_cache_key` before strict request validation:

```toml
[local]
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
model = "Qwen/Qwen2.5-Coder-32B-Instruct"
lmcache_passthrough = true
```

Equivalent environment override:

```bash
TINYCTX_LOCAL_LMCACHE_PASSTHROUGH=1
```

When disabled, tinyctx keeps the previous default and strips
`prompt_cache_key` for strict local backends such as LMStudio.

