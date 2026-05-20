from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app(tmp_path: Path) -> FastAPI:
    from tinyctx import dashboard

    app = FastAPI()
    dashboard.register(app, tmp_path)
    return app


def test_config_page_is_mounted(tmp_path: Path):
    client = TestClient(_make_app(tmp_path))

    r = client.get("/dashboard/config")

    assert r.status_code == 200
    assert "tinyctx Config Center" in r.text


def test_config_get_returns_raw_schema_and_effective_config(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[local]\nmodel = "demo"\n', encoding="utf-8")
    monkeypatch.setenv("TINYCTX_CONFIG", str(cfg))
    client = TestClient(_make_app(tmp_path))

    r = client.get("/api/v1/config")

    assert r.status_code == 200
    data = r.json()
    assert data["path"] == str(cfg)
    assert 'model = "demo"' in data["raw"]
    assert "schema" in data
    assert data["effective"]["local"]["model"] == "demo"


def test_config_validate_reports_errors(tmp_path: Path):
    client = TestClient(_make_app(tmp_path))

    r = client.post("/api/v1/config/validate", json={
        "sections": {"local": {"base_url": "bad-url", "wire_api": "bad"}}
    })

    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert data["errors"]


def test_config_save_requires_local_write_opt_in(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("TINYCTX_CONFIG", str(cfg))
    client = TestClient(_make_app(tmp_path))

    blocked = client.post("/api/v1/config/save", json={
        "sections": {"local": {"model": "blocked"}}
    })
    assert blocked.status_code == 403

    monkeypatch.setenv("TINYCTX_DASHBOARD_WRITE", "1")
    saved = client.post("/api/v1/config/save", json={
        "sections": {"local": {"model": "saved"}}
    })
    assert saved.status_code == 200
    assert saved.json()["needs_restart"] is True
    assert 'model = "saved"' in cfg.read_text(encoding="utf-8")


def test_config_test_local_uses_submitted_sections(tmp_path: Path, monkeypatch):
    from tinyctx import dashboard

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            assert url == "http://lmstudio.test/v1/models"
            return _Resp(200, {"data": [{"id": "demo"}]})

        def post(self, url, json=None, headers=None):
            assert url == "http://lmstudio.test/v1/chat/completions"
            assert headers["Authorization"] == "Bearer lm-studio"
            assert json["model"] == "demo"
            return _Resp(200, {"choices": [{"message": {"content": "pong"}}]})

    monkeypatch.setattr(dashboard.httpx, "Client", _Client)
    client = TestClient(_make_app(tmp_path))

    r = client.post("/api/v1/config/test-local", json={
        "sections": {
            "local": {
                "base_url": "http://lmstudio.test/v1",
                "wire_api": "chat",
                "model": "demo",
                "headers": {"Authorization": "Bearer lm-studio"},
            }
        }
    })

    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["models_status"] == 200
    assert data["completion_status"] == 200
