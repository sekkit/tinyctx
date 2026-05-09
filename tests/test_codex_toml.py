"""Tests for tinyctx._codex_toml — the shared config.toml patcher.

Covers:
  - exact line-of-its-own marker matching (not naive substring)
  - lock-protected append eliminates race-induced duplicates
  - strip_mcp_block leaves siblings untouched
  - has_mcp_block uses the same line-exact rule
"""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from tinyctx._codex_toml import (
    _marker_present,
    append_mcp_block,
    has_mcp_block,
    strip_mcp_block,
)


# ─────────── exact-line matching ───────────


def test_marker_present_matches_exact_line():
    text = "[mcp_servers.gitnexus]\ntype = \"stdio\"\n"
    assert _marker_present(text, "[mcp_servers.gitnexus]") is True


def test_marker_present_ignores_substring_in_comment():
    """Naive substring match would trip on a comment that mentions the
    section name (e.g. `# see [mcp_servers.gitnexus] for details`).
    Line-exact matching avoids that false positive."""
    text = "# refer to [mcp_servers.gitnexus] in docs\nfoo = 1\n"
    assert _marker_present(text, "[mcp_servers.gitnexus]") is False


def test_marker_present_with_leading_whitespace_still_matches():
    """A real TOML header may be indented (rare but legal)."""
    text = "  [mcp_servers.gitnexus]\ntype = \"stdio\"\n"
    assert _marker_present(text, "[mcp_servers.gitnexus]") is True


def test_marker_present_with_trailing_whitespace_still_matches():
    text = "[mcp_servers.gitnexus]   \ntype = \"stdio\"\n"
    assert _marker_present(text, "[mcp_servers.gitnexus]") is True


def test_marker_present_does_not_match_subsection():
    """[mcp_servers.advisor] is NOT [mcp_servers.advisor.env]."""
    text = "[mcp_servers.advisor.env]\nKEY = \"x\"\n"
    assert _marker_present(text, "[mcp_servers.advisor]") is False


# ─────────── append idempotency ───────────


def test_append_creates_new_file(tmp_path):
    p = tmp_path / "codex" / "config.toml"
    block = "\n[mcp_servers.foo]\ntype = \"stdio\"\n"
    ok, _ = append_mcp_block(p, "[mcp_servers.foo]", block)
    assert ok and p.is_file()
    assert "[mcp_servers.foo]" in p.read_text()


def test_append_idempotent(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[mcp_servers.foo]\ntype = \"stdio\"\n")
    snap = p.read_text()
    ok, msg = append_mcp_block(p, "[mcp_servers.foo]",
                               "[mcp_servers.foo]\nx = 1\n")
    assert ok
    assert "already configured" in msg
    assert p.read_text() == snap


def test_append_dry_run_does_not_write(tmp_path):
    p = tmp_path / "config.toml"
    ok, msg = append_mcp_block(p, "[mcp_servers.foo]", "block",
                               dry_run=True)
    assert ok
    assert "DRY-RUN" in msg
    assert not p.exists()


def test_append_does_not_match_substring_in_comment(tmp_path):
    """Real-world bug: user had `# see [mcp_servers.foo] above` in a
    comment. Old substring check would think the section was present
    and skip writing. New line-exact check writes it."""
    p = tmp_path / "config.toml"
    p.write_text("# see [mcp_servers.foo] for details\nbar = 1\n")
    ok, msg = append_mcp_block(p, "[mcp_servers.foo]",
                               "\n[mcp_servers.foo]\nx = 1\n")
    assert ok and "appended" in msg
    text = p.read_text()
    # Must contain BOTH the comment AND the new section
    assert "# see [mcp_servers.foo] for details" in text
    section_lines = [l for l in text.splitlines()
                     if l.strip() == "[mcp_servers.foo]"]
    assert len(section_lines) == 1


def test_has_mcp_block_uses_line_exact_rule(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("# documentation: [mcp_servers.x] is the foo MCP\n")
    assert has_mcp_block(p, "[mcp_servers.x]") is False
    p.write_text("[mcp_servers.x]\ntype=\"stdio\"\n")
    assert has_mcp_block(p, "[mcp_servers.x]") is True


# ─────────── race / lock test ───────────


def _race_worker(args):
    """Worker for parallel append. Returns (ok, msg)."""
    path_str, marker, block = args
    return append_mcp_block(Path(path_str), marker, block)


def test_concurrent_appends_do_not_duplicate(tmp_path):
    """5 parallel processes try to append the same marker. With flock,
    exactly one writes; the rest see "already configured". WITHOUT the
    lock this test would intermittently produce 2-5 copies of the
    section header (the original bug observed in production)."""
    p = tmp_path / "config.toml"
    p.write_text("# initial\n")
    marker = "[mcp_servers.race]"
    block = "\n[mcp_servers.race]\ntype = \"stdio\"\n"
    args = [(str(p), marker, block) for _ in range(5)]
    with multiprocessing.Pool(5) as pool:
        results = pool.map(_race_worker, args)

    # All processes return ok=True.
    assert all(ok for ok, _ in results)

    text = p.read_text()
    section_lines = [l for l in text.splitlines() if l.strip() == marker]
    assert len(section_lines) == 1, (
        f"expected exactly 1 [mcp_servers.race] header, "
        f"found {len(section_lines)}\n--- file ---\n{text}")


# ─────────── strip ───────────


def test_strip_removes_block_and_leading_comments():
    text = (
        "[mcp_servers.advisor]\n"
        'cmd = "x"\n'
        "\n"
        "# Added by tinyctx (gitnexus_bootstrap)\n"
        "[mcp_servers.gitnexus]\n"
        'type = "stdio"\n'
        'command = "g"\n'
        "\n"
        "[mcp_servers.serena]\n"
        'cmd = "s"\n'
    )
    out = strip_mcp_block(text, "[mcp_servers.gitnexus]")
    # gitnexus block + its comment header gone
    assert "[mcp_servers.gitnexus]" not in out
    assert "# Added by tinyctx" not in out
    # neighbours preserved
    assert "[mcp_servers.advisor]" in out
    assert "[mcp_servers.serena]" in out


def test_strip_no_op_when_marker_absent():
    text = "[mcp_servers.advisor]\ncmd = \"x\"\n"
    out = strip_mcp_block(text, "[mcp_servers.gitnexus]")
    assert out == text


def test_strip_block_at_eof():
    text = (
        "[mcp_servers.advisor]\n"
        'cmd = "x"\n'
        "\n"
        "# Added by tinyctx\n"
        "[mcp_servers.last]\n"
        'cmd = "y"\n'
    )
    out = strip_mcp_block(text, "[mcp_servers.last]")
    assert "[mcp_servers.last]" not in out
    assert "[mcp_servers.advisor]" in out


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if fn.__code__.co_argcount > 0:
                    # need tmp_path; skip in __main__ runner
                    continue
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(failed)
