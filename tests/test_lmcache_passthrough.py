from __future__ import annotations

import asyncio
import json

from tinyctx.config import (
    BackendCfg,
    Config,
    effective_strip_request_fields,
    load_config,
)
from tinyctx.sanitize import strip_unsupported_responses_fields


def _sanitize_for_backend(body: dict, backend: BackendCfg) -> dict:
    return strip_unsupported_responses_fields(
        body,
        drop=effective_strip_request_fields(backend),
    )


def test_default_backend_still_strips_prompt_cache_key():
    backend = BackendCfg(base_url="http://127.0.0.1:1234/v1")
    out = _sanitize_for_backend(
        {"prompt_cache_key": "repo-session", "client_metadata": {"x": 1}},
        backend,
    )
    assert "prompt_cache_key" not in out
    assert "client_metadata" not in out


def test_lmcache_passthrough_preserves_prompt_cache_key_only():
    backend = BackendCfg(
        base_url="http://127.0.0.1:8000/v1",
        lmcache_passthrough=True,
    )
    out = _sanitize_for_backend(
        {"prompt_cache_key": "repo-session", "client_metadata": {"x": 1}},
        backend,
    )
    assert out["prompt_cache_key"] == "repo-session"
    assert "client_metadata" not in out


def test_lmcache_passthrough_leaves_unrelated_strip_fields_unchanged():
    backend = BackendCfg(
        base_url="https://example.invalid",
        strip_request_fields=("max_output_tokens",),
        lmcache_passthrough=True,
    )
    assert effective_strip_request_fields(backend) == ("max_output_tokens",)


def test_load_config_reads_local_lmcache_passthrough_from_toml(
    monkeypatch,
    tmp_path,
):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[local]\n"
        "lmcache_passthrough = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TINYCTX_CONFIG", str(cfg_path))

    cfg = load_config()

    assert cfg.local.lmcache_passthrough is True


def test_load_config_reads_local_lmcache_passthrough_from_env(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TINYCTX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("TINYCTX_LOCAL_LMCACHE_PASSTHROUGH", "1")

    cfg = load_config()

    assert cfg.local.lmcache_passthrough is True


def test_proxy_preserves_prompt_cache_key_when_local_lmcache_enabled(
    monkeypatch,
):
    from tinyctx import proxy

    cfg = Config()
    cfg.force_route = "local"
    cfg.local = BackendCfg(
        base_url="http://local.invalid/v1",
        wire_api="responses",
        lmcache_passthrough=True,
        inject_defaults={},
        cap_fields={},
    )
    cfg.inject_global_agent_rules = False
    cfg.auto_scout = False
    monkeypatch.setattr(proxy, "CFG", cfg)
    proxy._SESSION_ERROR_STREAK.clear()

    captured: dict[str, object] = {}

    async def fake_forward(url, headers, body, is_stream, sid, decision, **kwargs):
        captured["url"] = url
        captured["body"] = body
        return {"ok": True}

    monkeypatch.setattr(proxy, "_forward", fake_forward)

    class Request:
        headers = {}

        async def body(self):
            return json.dumps({
                "model": "gpt-5.5",
                "prompt_cache_key": "repo-session",
                "client_metadata": {"drop": True},
                "input": [{"role": "user", "content": "hi"}],
            }).encode()

    result = asyncio.run(proxy.responses(Request()))

    assert result == {"ok": True}
    assert captured["body"]["prompt_cache_key"] == "repo-session"
    assert "client_metadata" not in captured["body"]
