"""Tests for tinyctx.historian: rolling per-session compression + on-wire
substitution + idempotence."""
from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from tinyctx import historian
from tinyctx.config import BackendCfg


def _backend() -> BackendCfg:
    return BackendCfg(base_url="http://127.0.0.1:1/v1",
                      model="qwen-fake", wire_api="chat", timeout_s=10.0)


def _body(n_turns: int) -> dict:
    items = []
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        items.append({"role": role, "content": f"turn-{i}-content " * 50})
    return {"input": items}


# --------------------------------------------------------- update half


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _fake_llm_ok(client, backend, system_prompt, user_prompt, **kw):
    return (
        "## What we are doing and why\nbuilding a feature\n"
        "## Files & decisions\nsrc/x.py\n"
        "## Commands & outcomes\npytest passed\n"
        "## Open issues / next steps\nreview pending\n\n"
        '```json\n{"compartments": [{"name": "feat-x", "topic": "Feature X", '
        '"summary": "S", "files": ["src/x.py"]}], '
        '"facts": [{"claim": "x is wired", "evidence": "logs"}], '
        '"open_questions": ["q1"]}\n```'
    )


async def _fake_llm_fail(*a, **kw):
    raise RuntimeError("simulated LLM failure")


def test_update_skips_when_not_enough_new_turns():
    historian.reset_session("s1")
    with TemporaryDirectory() as td:
        proj = Path(td) / "proj"; proj.mkdir()
        body = _body(6)  # 6 turns, recent_keep=4 → only 2 to compress
        # default min_new_turns=5; first time no last run → 6-0 >= 5 OK
        # but we test the OPPOSITE: set last_run to a high number.
        historian.get_state("s1").last_run_turn_count = 6
        ok = _run(historian.update("s1", body, _backend(),
                                   project_root=proj,
                                   _llm_call=_fake_llm_ok))
        assert ok is False  # no new turns since last run


def test_update_skips_when_history_smaller_than_recent_keep():
    historian.reset_session("s2")
    with TemporaryDirectory() as td:
        proj = Path(td) / "proj"; proj.mkdir()
        ok = _run(historian.update("s2", _body(3), _backend(),
                                   recent_keep=4, project_root=proj,
                                   _llm_call=_fake_llm_ok))
        assert ok is False


def test_update_writes_digest_files_when_triggered():
    historian.reset_session("s3")
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"; proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            ok = _run(historian.update("s3", _body(20), _backend(),
                                       min_new_turns=5, recent_keep=4,
                                       project_root=proj,
                                       _llm_call=_fake_llm_ok))
            assert ok is True
            sdir = historian.historian_dir(proj, "s3")
            assert (sdir / "historian-1.md").is_file()
            assert (sdir / "historian-1.json").is_file()
            data = json.loads((sdir / "historian-1.json").read_text())
            assert data["facts"][0]["claim"] == "x is wired"
            state = historian.get_state("s3")
            assert state.revision == 1
            assert state.last_run_turn_count == 20  # _count_history_items


def test_update_increments_revision_each_pass():
    historian.reset_session("s4")
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"; proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            _run(historian.update("s4", _body(10), _backend(),
                                  min_new_turns=5, project_root=proj,
                                  _llm_call=_fake_llm_ok))
            _run(historian.update("s4", _body(20), _backend(),
                                  min_new_turns=5, project_root=proj,
                                  _llm_call=_fake_llm_ok))
            sdir = historian.historian_dir(proj, "s4")
            assert (sdir / "historian-1.md").is_file()
            assert (sdir / "historian-2.md").is_file()
            assert historian.get_state("s4").revision == 2


def test_update_quiet_on_llm_failure():
    historian.reset_session("s5")
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"; proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            ok = _run(historian.update("s5", _body(10), _backend(),
                                       min_new_turns=5, project_root=proj,
                                       _llm_call=_fake_llm_fail))
            assert ok is False
            assert historian.get_state("s5").revision == 0
            assert not (historian.historian_dir(proj, "s5")).exists()


# --------------------------------------------------------- apply half


def test_apply_no_op_without_digest():
    historian.reset_session("s6")
    body = _body(10)
    out = historian.apply_to_body(body, "s6")
    assert out == body  # untouched


def test_apply_substitutes_old_turns_when_digest_present():
    historian.reset_session("s7")
    state = historian.get_state("s7")
    state.last_digest_md = "DIGEST-CONTENT"
    state.revision = 1
    body = _body(10)
    out = historian.apply_to_body(body, "s7", recent_keep=4)
    items = out["input"]
    # First item is the digest system message.
    assert items[0]["role"] == "system"
    assert "DIGEST-CONTENT" in items[0]["content"]
    assert "<tinyctx-historian-digest" in items[0]["content"]
    # The trailing 4 turns are preserved verbatim.
    assert len(items) == 5  # 1 digest + 4 recent
    assert items[1] == body["input"][6]   # 10-4 = 6


def test_apply_is_idempotent():
    historian.reset_session("s8")
    state = historian.get_state("s8")
    state.last_digest_md = "DIGEST-CONTENT"
    state.revision = 3
    body = _body(10)
    out1 = historian.apply_to_body(body, "s8", recent_keep=4)
    # Running again on the already-substituted body must not nest digests.
    out2 = historian.apply_to_body(out1, "s8", recent_keep=4)
    assert out1 == out2


def test_apply_does_not_mutate_input():
    historian.reset_session("s9")
    state = historian.get_state("s9")
    state.last_digest_md = "DIGEST"
    body = _body(10)
    snapshot = deepcopy(body)
    _ = historian.apply_to_body(body, "s9")
    assert body == snapshot


def test_spawn_update_returns_a_task_object():
    """Smoke: spawn_update returns a Task; the strong reference is held in
    historian._BG_TASKS until completion."""
    historian.reset_session("s10")
    with TemporaryDirectory() as td:
        td_p = Path(td)
        proj = td_p / "proj"; proj.mkdir()
        with mock.patch.dict(os.environ, {"HOME": str(td_p), "USERPROFILE": str(td_p)}):
            async def _go():
                t = historian.spawn_update("s10", _body(10), _backend(),
                                           min_new_turns=5, project_root=proj)
                assert t in historian._BG_TASKS
                # Don't actually wait for the LLM call; cancel it.
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            asyncio.new_event_loop().run_until_complete(_go())


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
