"""Shared FastAPI dependencies. Endpoints declare these as type annotations,
e.g. `async def handler(db: DbSession, risk: RiskModel)`."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.hazard_detector import HazardDetectorService
from app.services.risk_model import RiskModelService


def get_risk_model(request: Request) -> RiskModelService:
    return request.app.state.risk_model


def get_hazard_detector(request: Request) -> HazardDetectorService:
    return request.app.state.hazard_detector


DbSession = Annotated[AsyncSession, Depends(get_db)]
RiskModel = Annotated[RiskModelService, Depends(get_risk_model)]
HazardDetector = Annotated[HazardDetectorService, Depends(get_hazard_detector)]
