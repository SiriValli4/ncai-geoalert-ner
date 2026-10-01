from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.api.deps import DbSession, HazardDetector, RiskModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    """Process is up. Never touches the DB, so orchestrators don't restart a
    healthy pod just because Postgres is briefly unreachable."""
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness(
    response: Response, db: DbSession, risk: RiskModel, detector: HazardDetector
) -> dict:
    """Ready = database + PostGIS reachable. ML models are reported but are
    not required: their endpoints return 503 individually when missing."""
    db_status: dict[str, object]
    try:
        postgis = await db.scalar(text("SELECT postgis_lib_version()"))
        db_status = {"ok": True, "postgis_version": postgis}
    except Exception as exc:
        logger.warning("Readiness DB check failed: %s", exc)
        db_status = {"ok": False, "error": type(exc).__name__}
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if db_status["ok"] else "unavailable",
        "database": db_status,
        "models": {
            "risk": {
                "loaded": risk.is_loaded,
                "model_type": risk.model_type or None,
                "shap_available": risk.shap_explainer is not None,
            },
            "hazard_detector": {
                "loaded": detector.is_loaded,
                "classes": list(detector.class_names.values()),
                "heic_support": detector.heic_support,
            },
        },
    }
