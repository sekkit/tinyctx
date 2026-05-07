"""Tests for tinyctx.memory. We don't require mem0 to be installed; these
tests verify graceful degradation. The "live" mem0 path is only exercised
when the user opts into the [mem] extra."""
from __future__ import annotations

import io
import contextlib
import sys
from unittest import mock

from tinyctx import memory


def test_is_available_matches_mem0_importability():
    """Manual oracle: tinyctx.memory.is_available() is True iff `import
    mem0` succeeds. Most CI environments don't have mem0 → False."""
    try:
        import mem0  # noqa: F401
        assert memory.is_available() is True
    except ImportError:
        assert memory.is_available() is False


def test_memstore_raises_clean_error_when_mem0_absent():
    """If mem0 isn't installed, MemStore() raises ImportError with a
    helpful message instead of crashing somewhere weird."""
    if memory.is_available():
        # mem0 IS installed; we can't test the absent path without faking it.
        return
    try:
        memory.MemStore()
    except ImportError as e:
        assert "tinyctx[mem]" in str(e) or "mem0ai" in str(e)
        return
    raise AssertionError("MemStore() did not raise ImportError")


def test_cli_available_subcommand():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = memory.main(["available"])
    out = buf.getvalue().strip()
    if memory.is_available():
        assert rc == 0
        assert out == "yes"
    else:
        assert rc == 1
        assert out.startswith("no")


def test_cli_add_without_mem0_returns_error():
    """`tinyctx-mem add ...` should not crash, just exit 1."""
    if memory.is_available():
        return
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        rc = memory.main(["add", "user prefers tabs"])
    assert rc == 1
    assert "not installed" in buf_err.getvalue()


def test_cli_search_without_mem0_returns_error():
    if memory.is_available():
        return
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        rc = memory.main(["search", "tabs"])
    assert rc == 1
    assert "not installed" in buf_err.getvalue()


def test_ingest_compaction_without_structured_returns_error():
    """If there's no structured compaction yet, ingest must fail cleanly."""
    if not memory.is_available():
        # Without mem0, we still hit the "not installed" path first.
        buf_err = io.StringIO()
        with contextlib.redirect_stderr(buf_err):
            rc = memory.main(["ingest-compaction", "--root", "/tmp"])
        assert rc == 1
        return
    # If mem0 IS installed, /tmp won't have a structured compaction.
    buf_err = io.StringIO()
    with contextlib.redirect_stderr(buf_err):
        rc = memory.main(["ingest-compaction", "--root", "/tmp"])
    assert rc == 1


if __name__ == "__main__":
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
