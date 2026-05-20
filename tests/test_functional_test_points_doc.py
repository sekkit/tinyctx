from __future__ import annotations

from pathlib import Path


def test_functional_test_points_doc_lists_every_test_file():
    root = Path(__file__).resolve().parents[1]
    doc = root / "docs" / "functional-test-points.md"
    text = doc.read_text(encoding="utf-8")

    test_files = sorted(
        path.name
        for path in (root / "tests").glob("test_*.py")
        if path.name != "test_functional_test_points_doc.py"
    )

    missing = [name for name in test_files if f"`tests/{name}`" not in text]
    assert not missing, "Document missing test files: " + ", ".join(missing)


def test_functional_test_points_doc_has_required_sections():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "functional-test-points.md").read_text(
        encoding="utf-8"
    )

    for heading in (
        "## Routing and Proxy",
        "## Context Compression and Continuity",
        "## Stream Reliability and Recovery",
        "## Bootstrap and Integrations",
        "## Observability and Operations",
        "## Configuration and Safety",
    ):
        assert heading in text

