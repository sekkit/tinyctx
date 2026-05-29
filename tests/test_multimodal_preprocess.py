"""Tests for `multimodal_preprocess` — the gated image→text caption hook.

mm CLI is mocked via a subprocess.run patch so tests do not depend on
having `mm` actually installed.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tinyctx import multimodal_preprocess as mmp


# Tiny 1×1 PNG (transparent) — enough to give `mm` a real file.
PNG_BYTES = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


def _body_with_image(image_url: str = PNG_DATA_URI) -> dict:
    return {
        "model": "tinyctx-auto",
        "input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {"type": "input_image", "image_url": image_url},
            ]},
        ],
    }


def _fake_mm_run(caption: str, returncode: int = 0):
    """Build a subprocess.run side_effect that pretends to be mm."""
    def _run(*args, **kwargs):
        payload = json.dumps({"caption": caption})
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=returncode,
            stdout=payload if returncode == 0 else "",
            stderr="" if returncode == 0 else "mm: failure",
        )
    return _run


def test_disabled_passthrough():
    body = _body_with_image()
    out, stats = mmp.preprocess(body, enabled=False)
    assert out is body
    assert stats == {"enabled": False, "images_seen": 0, "images_captioned": 0,
                     "cache_hits": 0, "errors": 0, "skipped_remote_url": 0}


def test_missing_mm_binary_passthrough(monkeypatch):
    body = _body_with_image()
    monkeypatch.setattr(mmp, "_which_mm", lambda: "")
    out, stats = mmp.preprocess(body, enabled=True)
    assert out is body
    assert stats["errors"] == 1
    assert stats.get("error_reason") == "mm binary not found"


def test_caption_replaces_image_and_caches(monkeypatch, tmp_path):
    body = _body_with_image()
    monkeypatch.setattr(mmp, "_which_mm", lambda: "/fake/mm")
    fake = _fake_mm_run("A tiny transparent test pixel.")
    with patch("tinyctx.multimodal_preprocess.subprocess.run", side_effect=fake):
        out, stats = mmp.preprocess(
            body, enabled=True, mm_bin="/fake/mm",
            cache_dir=tmp_path, timeout_s=5.0,
        )
    assert stats["images_seen"] == 1
    assert stats["images_captioned"] == 1
    assert stats["cache_hits"] == 0
    assert stats["errors"] == 0
    # Image item should be gone, replaced by an input_text marker.
    content = out["input"][0]["content"]
    types = [c["type"] for c in content]
    assert "input_image" not in types
    assert types.count("input_text") == 2  # original text + caption marker
    caption_item = content[-1]
    assert "caption: A tiny transparent test pixel." in caption_item["text"]
    assert "sha256=" in caption_item["text"]

    # Cache file landed on disk.
    cached_files = list(tmp_path.rglob("*.txt"))
    assert len(cached_files) == 1
    assert "A tiny transparent" in cached_files[0].read_text()


def test_cache_hit_avoids_mm_call(monkeypatch, tmp_path):
    body = _body_with_image()
    monkeypatch.setattr(mmp, "_which_mm", lambda: "/fake/mm")

    # Prime cache by running once.
    fake = _fake_mm_run("Cached caption text.")
    with patch("tinyctx.multimodal_preprocess.subprocess.run", side_effect=fake) as mock_run:
        mmp.preprocess(body, enabled=True, mm_bin="/fake/mm",
                       cache_dir=tmp_path, timeout_s=5.0)
        first_call_count = mock_run.call_count

    # Second pass — mm should NOT be invoked because cache hits.
    with patch("tinyctx.multimodal_preprocess.subprocess.run") as mock_run2:
        out2, stats2 = mmp.preprocess(
            body, enabled=True, mm_bin="/fake/mm",
            cache_dir=tmp_path, timeout_s=5.0)
        assert mock_run2.call_count == 0
    assert stats2["cache_hits"] == 1
    assert stats2["images_captioned"] == 1
    assert "Cached caption text." in out2["input"][0]["content"][-1]["text"]
    assert first_call_count == 1  # mm was called exactly once on first pass


def test_remote_url_is_passthrough(monkeypatch, tmp_path):
    body = _body_with_image(image_url="https://example.com/x.png")
    monkeypatch.setattr(mmp, "_which_mm", lambda: "/fake/mm")
    out, stats = mmp.preprocess(
        body, enabled=True, mm_bin="/fake/mm",
        cache_dir=tmp_path, timeout_s=5.0,
    )
    assert stats["images_seen"] == 1
    assert stats["skipped_remote_url"] == 1
    assert stats["images_captioned"] == 0
    # Original image_url item still in place.
    types = [c["type"] for c in out["input"][0]["content"]]
    assert "input_image" in types


def test_mm_failure_keeps_image_in_place(monkeypatch, tmp_path):
    body = _body_with_image()
    monkeypatch.setattr(mmp, "_which_mm", lambda: "/fake/mm")
    fake = _fake_mm_run("", returncode=1)
    with patch("tinyctx.multimodal_preprocess.subprocess.run", side_effect=fake):
        out, stats = mmp.preprocess(
            body, enabled=True, mm_bin="/fake/mm",
            cache_dir=tmp_path, timeout_s=5.0,
        )
    assert stats["errors"] == 1
    assert stats["images_captioned"] == 0
    # Image still present for the downstream image_detected -> frontier rule.
    types = [c["type"] for c in out["input"][0]["content"]]
    assert "input_image" in types


def test_no_images_in_body_is_noop(monkeypatch, tmp_path):
    body = {"input": [{"role": "user", "content": [
        {"type": "input_text", "text": "plain text only"}]}]}
    monkeypatch.setattr(mmp, "_which_mm", lambda: "/fake/mm")
    with patch("tinyctx.multimodal_preprocess.subprocess.run") as mock_run:
        out, stats = mmp.preprocess(
            body, enabled=True, mm_bin="/fake/mm",
            cache_dir=tmp_path, timeout_s=5.0,
        )
    assert mock_run.call_count == 0
    assert stats["images_seen"] == 0
    assert stats["images_captioned"] == 0
    assert out is body  # short-circuit returns the original


def test_should_caption_images_locally_allows_low_risk_ocr():
    body = _body_with_image()
    body["input"][0]["content"][0]["text"] = "请提取图片中的文字并总结一下"

    assert mmp.should_caption_images_locally(body) is True


def test_should_caption_images_locally_keeps_accuracy_sensitive_screenshots_on_frontier():
    body = _body_with_image()
    body["input"][0]["content"][0]["text"] = (
        "看这个 UI 截图，按钮布局和颜色哪里不对？"
    )

    assert mmp.should_caption_images_locally(body) is False


def test_should_caption_images_locally_requires_explicit_low_risk_request():
    body = _body_with_image()
    body["input"][0]["content"][0]["text"] = "帮我看看这个图片"

    assert mmp.should_caption_images_locally(body) is False
