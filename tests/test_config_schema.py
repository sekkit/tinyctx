def test_validate_sections_accepts_chatgpt_frontier_without_openai_key():
    from tinyctx.config_schema import validate_sections

    result = validate_sections({
        "frontier": {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "wire_api": "responses",
            "model": "gpt-5.5",
        }
    })

    assert result["ok"] is True
    assert result["errors"] == []


def test_validate_sections_rejects_bad_url_and_wire_api():
    from tinyctx.config_schema import validate_sections

    result = validate_sections({
        "local": {
            "base_url": "ftp://bad",
            "wire_api": "socket",
        },
        "routing": {"force_route": "somewhere"},
    })

    assert result["ok"] is False
    messages = " ".join(e["message"] for e in result["errors"])
    assert "http:// or https://" in messages
    assert "chat or responses" in messages
    assert "auto, local, or frontier" in messages


def test_presets_include_lmstudio_and_codex_official():
    from tinyctx.config_schema import config_presets

    presets = config_presets()

    assert "lmstudio" in presets
    assert presets["lmstudio"]["sections"]["local"]["wire_api"] == "chat"
    assert "codex-official" in presets
    frontier = presets["codex-official"]["sections"]["frontier"]
    assert frontier["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert "api_key_env" not in frontier
