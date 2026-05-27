"""Regression tests for `proxy._extract_prompt_cache_usage` covering the
three upstream usage shapes tinyctx must understand:

* DeepSeek / OpenAI chat-completions: ``prompt_cache_hit_tokens`` +
  ``prompt_cache_miss_tokens`` (the original format)
* OpenAI Responses (codex frontier): ``input_tokens_details.cached_tokens``
* Anthropic Messages: ``cache_read_input_tokens`` +
  ``cache_creation_input_tokens``

Until 2026-05-26 the extractor only recognized DeepSeek's keys, so the
codex frontier reported 0/0 even when ~32% of input tokens were cached
(confirmed via forensic dump 20260526-161107). Both the dict-based
extractor and the regex-based buffer extractor are covered.
"""
from __future__ import annotations

from tinyctx.proxy import (
    _extract_prompt_cache_usage,
    _extract_prompt_cache_usage_from_buffer,
)


def test_deepseek_shape():
    assert _extract_prompt_cache_usage(
        {"usage": {"prompt_cache_hit_tokens": 50,
                   "prompt_cache_miss_tokens": 100}}
    ) == (50, 100)


def test_openai_responses_shape():
    # Real shape from forensic dump 20260526-161107-punt_via_stream_rewrite
    payload = {"usage": {
        "input_tokens": 160311,
        "input_tokens_details": {"cached_tokens": 50816},
        "output_tokens": 190,
        "total_tokens": 160501,
    }}
    assert _extract_prompt_cache_usage(payload) == (50816, 109495)


def test_openai_responses_nested_under_response_key():
    payload = {"response": {"usage": {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 250},
    }}}
    assert _extract_prompt_cache_usage(payload) == (250, 750)


def test_openai_chat_prompt_tokens_details():
    payload = {"usage": {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cached_tokens": 600},
        "completion_tokens": 200,
    }}
    assert _extract_prompt_cache_usage(payload) == (600, 400)


def test_anthropic_messages_shape():
    payload = {"usage": {
        "input_tokens": 50,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 200,
        "output_tokens": 30,
    }}
    # hit = cache_read; miss = uncached input + write-on-miss
    assert _extract_prompt_cache_usage(payload) == (200, 60)


def test_deepseek_wins_over_responses_when_both_present():
    # Backward compat: explicit DeepSeek fields take priority.
    payload = {"usage": {
        "prompt_cache_hit_tokens": 5,
        "prompt_cache_miss_tokens": 7,
        "input_tokens_details": {"cached_tokens": 999},
    }}
    assert _extract_prompt_cache_usage(payload) == (5, 7)


def test_empty_and_none_payloads():
    assert _extract_prompt_cache_usage({}) == (0, 0)
    assert _extract_prompt_cache_usage(None) == (0, 0)
    assert _extract_prompt_cache_usage("not a dict") == (0, 0)
    assert _extract_prompt_cache_usage({"usage": "not a dict"}) == (0, 0)


def test_zero_cached_responses():
    # cached_tokens: 0 is a legitimate signal (cold cache); should still
    # be reported, with all input tokens marked as miss.
    payload = {"usage": {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 0},
    }}
    assert _extract_prompt_cache_usage(payload) == (0, 1000)


# ─── buffer (regex fallback) ───────────────────────────────────────────


def test_buffer_deepseek():
    raw = 'data: {"usage":{"prompt_cache_hit_tokens":50,"prompt_cache_miss_tokens":100}}'
    assert _extract_prompt_cache_usage_from_buffer(raw) == (50, 100)


def test_buffer_openai_responses_sse_tail():
    raw = (
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_xxx",'
        '"usage":{"input_tokens":160202,'
        '"input_tokens_details":{"cached_tokens":50816},'
        '"output_tokens":265}}}\n\n'
    )
    assert _extract_prompt_cache_usage_from_buffer(raw) == (50816, 109386)


def test_buffer_anthropic():
    raw = ('data: {"usage":{"input_tokens":50,'
           '"cache_creation_input_tokens":10,'
           '"cache_read_input_tokens":200}}')
    assert _extract_prompt_cache_usage_from_buffer(raw) == (200, 60)


def test_buffer_empty_or_no_usage():
    assert _extract_prompt_cache_usage_from_buffer("") == (0, 0)
    assert _extract_prompt_cache_usage_from_buffer("hello world no JSON") == (0, 0)
