"""Tests for the scout PostToolUse trigger.

Focused on provider-scoping: the hook path (`touch --require-scout`) must be a
no-op for repos tinyctx has never scouted, so a codex session on a non-tinyctx
provider is never affected (no trigger file, no cache-dir side effects).
"""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import scout, scout_trigger


def _isolate(home: Path):
    # scout.cache_dir() keys off Path.home(); pin HOME so the test never
    # touches the real ~/.tinyctx.
    return mock.patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)})


def _seed_scout_cache(proj: Path) -> None:
    scout.cache_dir(proj).mkdir(parents=True, exist_ok=True)
    scout.manifest_path(proj).write_text(json.dumps({"version": scout.CACHE_VERSION}))


def test_touch_require_scout_noops_without_manifest():
    with TemporaryDirectory() as td:
        home = Path(td) / "home"; home.mkdir()
        proj = Path(td) / "proj"; proj.mkdir()
        with _isolate(home):
            rc = scout_trigger.main(
                ["touch", "--root", str(proj), "--quiet", "--require-scout"])
            assert rc == 0
            # No scout cache => no trigger file and no cache-dir pollution.
            assert not scout_trigger.trigger_path(proj).exists()
            assert not scout.cache_dir(proj).exists()


def test_touch_require_scout_fires_with_manifest():
    with TemporaryDirectory() as td:
        home = Path(td) / "home"; home.mkdir()
        proj = Path(td) / "proj"; proj.mkdir()
        with _isolate(home):
            _seed_scout_cache(proj)
            rc = scout_trigger.main(
                ["touch", "--root", str(proj), "--quiet", "--require-scout"])
            assert rc == 0
            assert scout_trigger.trigger_path(proj).exists()


def test_plain_touch_always_fires():
    with TemporaryDirectory() as td:
        home = Path(td) / "home"; home.mkdir()
        proj = Path(td) / "proj"; proj.mkdir()
        with _isolate(home):
            # Without --require-scout, a manual touch always creates the trigger.
            rc = scout_trigger.main(["touch", "--root", str(proj), "--quiet"])
            assert rc == 0
            assert scout_trigger.trigger_path(proj).exists()
