from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as main_module
from app.core.config import Settings, settings

VERCEL = "https://geoalert-ner.vercel.app"


def _settings(**env: str) -> Settings:
    # _env_file=None: ignore the developer's local .env so tests are deterministic.
    return Settings(_env_file=None, **env)  # type: ignore[call-arg, arg-type]


def test_origins_parse_comma_list_and_strip_trailing_slash() -> None:
    s = _settings(CORS_ORIGINS=f"{VERCEL}/ , http://localhost:3000")
    assert s.CORS_ORIGINS == [VERCEL, "http://localhost:3000"]
    assert not s.cors_allow_all


def test_origins_parse_json_list() -> None:
    s = _settings(CORS_ORIGINS=f'["{VERCEL}"]')
    assert s.CORS_ORIGINS == [VERCEL]


def test_wildcard_allowed_locally() -> None:
    assert _settings(ENVIRONMENT="local", CORS_ORIGINS="*").cors_allow_all


@pytest.mark.parametrize("env", ["staging", "production"])
def test_wildcard_rejected_outside_local(env: str) -> None:
    with pytest.raises(ValidationError, match="only allowed when ENVIRONMENT=local"):
        _settings(ENVIRONMENT=env, CORS_ORIGINS="*")


def _client_with(monkeypatch: pytest.MonkeyPatch, origins: list[str]) -> TestClient:
    monkeypatch.setattr(settings, "CORS_ORIGINS", origins)
    return TestClient(main_module.create_app())


def test_explicit_origin_allowed_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with(monkeypatch, [VERCEL])
    r = client.get("/", headers={"Origin": VERCEL})
    assert r.headers["access-control-allow-origin"] == VERCEL
    assert r.headers["access-control-allow-credentials"] == "true"

    r = client.get("/", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_wildcard_never_sends_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with(monkeypatch, ["*"])
    r = client.options(
        f"{settings.API_PREFIX}/health",
        headers={"Origin": "https://anything.example", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in r.headers
