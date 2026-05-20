from pathlib import Path


def test_merge_sections_preserves_unknowns_and_appends_missing_keys():
    from tinyctx.config_io import merge_sections_into_toml

    raw = """# user comments survive
[local]
base_url = "http://old.local/v1"
unknown_local = 123

[extra]
keep = "yes"
"""

    merged = merge_sections_into_toml(raw, {
        "local": {
            "base_url": "http://new.local/v1",
            "headers": {"Authorization": "Bearer lm-studio"},
        },
        "routing": {"force_route": "auto"},
    })

    assert "# user comments survive" in merged
    assert 'base_url = "http://new.local/v1"' in merged
    assert "unknown_local = 123" in merged
    assert "[extra]" in merged and 'keep = "yes"' in merged
    assert 'headers = { Authorization = "Bearer lm-studio" }' in merged
    assert "[routing]" in merged
    assert 'force_route = "auto"' in merged


def test_save_config_text_writes_backup_and_replaces_atomically(tmp_path: Path):
    from tinyctx.config_io import save_config_text

    path = tmp_path / "config.toml"
    path.write_text('[local]\nmodel = "old"\n', encoding="utf-8")

    result = save_config_text('[local]\nmodel = "new"\n', path=path)

    assert path.read_text(encoding="utf-8") == '[local]\nmodel = "new"\n'
    assert result["path"] == str(path)
    assert result["backup_path"]
    assert Path(result["backup_path"]).read_text(encoding="utf-8") == '[local]\nmodel = "old"\n'
