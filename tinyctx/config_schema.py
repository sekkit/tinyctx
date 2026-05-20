"""Schema, presets, and validation for the visual config center."""
from __future__ import annotations

from typing import Any


ALLOWED_SECTIONS = {"server", "routing", "local", "frontier"}


def config_schema() -> dict[str, Any]:
    return {
        "sections": {
            "server": [
                _field("host", "Server host", "string"),
                _field("port", "Server port", "integer", minimum=1, maximum=65535),
                _field("verbose", "Verbose logging", "boolean"),
            ],
            "routing": [
                _field("force_route", "Route override", "enum", options=["auto", "local", "frontier"]),
                _field("redirect_compaction_to_local", "Route compaction to local", "boolean"),
                _field("sanitize_encrypted_content", "Strip encrypted reasoning payloads", "boolean"),
                _field("escalate_input_tokens", "Escalate token threshold", "integer", minimum=0),
                _field("escalate_turn_count", "Escalate turn threshold", "integer", minimum=0),
                _field("escalate_on_error_streak", "Escalate after tool error streak", "integer", minimum=0),
            ],
            "local": _backend_fields(local=True),
            "frontier": _backend_fields(local=False),
        }
    }


def config_presets() -> dict[str, Any]:
    return {
        "lmstudio": {
            "label": "LMStudio / vLLM local",
            "description": "Local-first OpenAI-compatible chat endpoint.",
            "sections": {
                "local": {
                    "base_url": "http://127.0.0.1:1234/v1",
                    "wire_api": "chat",
                    "model": "local-model",
                    "strip_tools": True,
                    "headers": {"Authorization": "Bearer lm-studio"},
                },
                "frontier": {
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "wire_api": "responses",
                    "model": "gpt-5.5",
                },
            },
        },
        "deepseek": {
            "label": "DeepSeek API local path",
            "description": "Cheap chat-completions API backend with API key in env.",
            "sections": {
                "local": {
                    "base_url": "https://api.deepseek.com/v1",
                    "wire_api": "chat",
                    "model": "deepseek-v4-flash",
                    "api_key_env": "DEEPSEEK_API_KEY",
                },
            },
        },
        "openrouter": {
            "label": "OpenRouter local path",
            "description": "OpenAI-compatible hosted model router.",
            "sections": {
                "local": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "wire_api": "chat",
                    "model": "qwen/qwen-2.5-coder-32b-instruct",
                    "api_key_env": "OPENROUTER_API_KEY",
                },
            },
        },
        "codex-official": {
            "label": "Codex official frontier",
            "description": "Use ChatGPT Codex backend; no OPENAI_API_KEY required.",
            "sections": {
                "frontier": {
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "wire_api": "responses",
                    "model": "gpt-5.5",
                },
            },
        },
    }


def validate_sections(sections: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(sections, dict):
        return {"ok": False, "errors": [_err("", "", "sections must be an object")], "warnings": []}

    for section, values in sections.items():
        if section not in ALLOWED_SECTIONS:
            warnings.append(_warn(section, "", "unknown section will be preserved but is not editable in the UI"))
            continue
        if not isinstance(values, dict):
            errors.append(_err(section, "", "section must be an object"))
            continue
        for key, value in values.items():
            _validate_field(errors, warnings, section, str(key), value)

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _validate_field(
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
    section: str,
    key: str,
    value: Any,
) -> None:
    if key == "base_url":
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            errors.append(_err(section, key, "base_url must start with http:// or https://"))
    elif key == "wire_api":
        if value not in ("chat", "responses"):
            errors.append(_err(section, key, "wire_api must be chat or responses"))
    elif key == "force_route":
        if value not in ("auto", "local", "frontier"):
            errors.append(_err(section, key, "force_route must be auto, local, or frontier"))
    elif key in {"port", "context_window", "timeout_s", "escalate_input_tokens", "escalate_turn_count", "escalate_on_error_streak"}:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(_err(section, key, f"{key} must be a non-negative integer"))
        elif key == "port" and value > 65535:
            errors.append(_err(section, key, "port must be between 1 and 65535"))
    elif key in {"verbose", "redirect_compaction_to_local", "sanitize_encrypted_content", "strip_tools", "lmcache_passthrough"}:
        if not isinstance(value, bool):
            errors.append(_err(section, key, f"{key} must be true or false"))
    elif key == "headers":
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            errors.append(_err(section, key, "headers must be a string-to-string object"))
    elif key in {"host", "model", "api_key_env"}:
        if value is not None and not isinstance(value, str):
            errors.append(_err(section, key, f"{key} must be a string"))
    else:
        warnings.append(_warn(section, key, "unknown key will be saved but is not validated"))


def _backend_fields(*, local: bool) -> list[dict[str, Any]]:
    fields = [
        _field("base_url", "Base URL", "url"),
        _field("wire_api", "Wire API", "enum", options=["chat", "responses"]),
        _field("model", "Model", "string"),
        _field("api_key_env", "API key env var", "string"),
        _field("context_window", "Context window", "integer", minimum=0),
        _field("timeout_s", "Timeout seconds", "integer", minimum=0),
        _field("headers", "Extra headers", "object"),
    ]
    if local:
        fields.extend([
            _field("strip_tools", "Strip tool definitions", "boolean"),
            _field("lmcache_passthrough", "LMCache headers passthrough", "boolean"),
        ])
    return fields


def _field(name: str, label: str, kind: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "label": label, "type": kind, **extra}


def _err(section: str, key: str, message: str) -> dict[str, str]:
    return {"section": section, "key": key, "message": message}


def _warn(section: str, key: str, message: str) -> dict[str, str]:
    return {"section": section, "key": key, "message": message}
