from __future__ import annotations

from tinyctx import skill_catalog


def test_default_catalog_contains_required_entries():
    catalog = skill_catalog.default_catalog()

    assert set(catalog) == {"skills", "mcp"}

    for name in ["cc-tdd", "cc-work", "cc-design", "huashu-design", "cc-research"]:
        entry = catalog["skills"][name]
        assert entry["kind"] == "skill"
        assert entry["description"]
        assert entry["task_types"]

    for name in ["context-mode", "browser", "gitnexus", "serena", "advisor"]:
        entry = catalog["mcp"][name]
        assert entry["kind"] == "mcp"
        assert entry["description"]
        assert entry["task_types"]


def test_summarize_catalog_respects_max_chars_and_mentions_groups():
    catalog = skill_catalog.default_catalog()

    full_summary = skill_catalog.summarize_catalog(catalog, max_chars=800)
    short_summary = skill_catalog.summarize_catalog(catalog, max_chars=120)

    assert "skills:" in full_summary
    assert "mcp:" in full_summary
    assert "cc-tdd" in full_summary
    assert "context-mode" in full_summary
    assert len(short_summary) <= 120
    assert short_summary.endswith("...") or len(full_summary) <= 120
