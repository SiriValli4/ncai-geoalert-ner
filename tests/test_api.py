from __future__ import annotations

import importlib.util

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings

API = settings.API_PREFIX


def test_liveness(client: TestClient) -> None:
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_lists_core_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for p in (
        f"{API}/risk/predict",
        f"{API}/hazards/verify-image",
        f"{API}/hazard-reports",
        f"{API}/sensors/nearby",
        f"{API}/risk-zones",
        "/api/submit-report",
    ):
        assert p in paths, p


def test_submit_report_rejects_unknown_hazard_type(client: TestClient) -> None:
    r = client.post(
        "/api/submit-report",
        data={"hazardType": "Volcano", "latitude": "25.5", "longitude": "91.8"},
    )
    assert r.status_code == 422
    assert "Unknown hazardType" in r.json()["detail"]


def test_submit_report_requires_location(client: TestClient) -> None:
    # The PWA sends "" when geolocation failed; that must be a clean 422.
    r = client.post(
        f"{API}/hazard-reports",
        data={"hazardType": "Rockfall", "latitude": "", "longitude": ""},
    )
    assert r.status_code == 422


def test_hazard_report_radius_params_must_be_together(client: TestClient) -> None:
    r = client.get(f"{API}/hazard-reports", params={"lat": 25.5})
    assert r.status_code == 422


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="vision extra installed; 503 path not reachable",
)
def test_verify_image_503_without_vision_stack(client: TestClient) -> None:
    r = client.post(
        f"{API}/hazards/verify-image",
        files={"file": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")},
    )
    assert r.status_code == 503


@pytest.mark.skipif(
    not settings.RISK_MODEL_PATH.exists() or importlib.util.find_spec("sklearn") is None,
    reason="risk model artifact or `risk` extra not available",
)
def test_predict_risk_with_real_model(client: TestClient) -> None:
    r = client.post(
        f"{API}/risk/predict",
        json={
            "latitude": 25.578,
            "longitude": 91.893,
            "rainfall_24h_mm": 210.5,
            "soil_moisture_percent": 78.0,
            "slope_degree": 42.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["landslide_susceptibility_index"] <= 100.0
    assert body["severity"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(body["top_factors"]) == 2
    assert set(body["estimated_inputs"]) == {
        "land_cover_code",
        "lithology_stability_index",
        "elevation_m",
    }


def test_predict_risk_validates_ranges(client: TestClient) -> None:
    r = client.post(
        f"{API}/risk/predict",
        json={
            "latitude": 25.5,
            "longitude": 91.8,
            "rainfall_24h_mm": -1,
            "soil_moisture_percent": 50,
            "slope_degree": 30,
        },
    )
    assert r.status_code == 422
