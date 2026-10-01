"""Application entry point.

uvicorn app.main:app --reload            # local dev
uvicorn app.main:app --host 0.0.0.0      # container (see Dockerfile)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints import hazards
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import engine
from app.services.hazard_detector import HazardDetectorService
from app.services.risk_model import RiskModelService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Models load once per process. A missing/broken model is logged and its
    # endpoints return 503; it never prevents the API from starting.
    app.state.risk_model = RiskModelService()
    app.state.risk_model.try_load(settings.RISK_MODEL_PATH)

    app.state.hazard_detector = HazardDetectorService(max_dim=settings.HAZARD_MAX_DIM)
    app.state.hazard_detector.try_load(settings.HAZARD_MODEL_PATH)

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    configure_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Landslide early-warning API for the North Eastern Region of India: "
        "risk prediction, AI photo verification, field reports, sensors and risk zones.",
        lifespan=lifespan,
        # Interactive docs are useful everywhere except a public production edge.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_PREFIX)

    # Backwards-compatible path the field PWA already posts to
    # (frontend/src/FieldReportPWA.jsx: SUBMIT_ENDPOINT = '/api/submit-report').
    app.add_api_route(
        "/api/submit-report",
        hazards.submit_report,
        methods=["POST"],
        status_code=201,
        response_model=hazards.HazardReportRead,
        tags=["hazard reports"],
        summary="Alias of POST {API_PREFIX}/hazard-reports for the field PWA",
        deprecated=True,
    )

    settings.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(settings.MEDIA_URL_PREFIX, StaticFiles(directory=settings.MEDIA_DIR), name="media")

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # Log the traceback server-side; never leak internals to clients.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "api": settings.API_PREFIX,
        }

    return app


app = create_app()
