from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy.exc import DBAPIError

from app.api.deps import DbSession
from app.models import SeverityLevel
from app.repositories import risk_zones as repo
from app.schemas.risk_zone import RiskZoneCreate, RiskZoneFeatureCollection, RiskZoneRead

router = APIRouter(prefix="/risk-zones", tags=["risk zones"])


@router.get(
    "",
    response_model=RiskZoneFeatureCollection,
    summary="Risk zones as a GeoJSON FeatureCollection (for the dashboard map)",
)
async def list_zones(
    db: DbSession,
    severity: SeverityLevel | None = None,
    limit: int = Query(1000, ge=1, le=10_000),
) -> RiskZoneFeatureCollection:
    zones = [RiskZoneRead.model_validate(z) for z in await repo.list_zones(db, severity, limit)]
    return RiskZoneFeatureCollection.from_zones(zones)


@router.post(
    "",
    response_model=RiskZoneRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a risk zone",
)
async def create_zone(payload: RiskZoneCreate, db: DbSession) -> RiskZoneRead:
    try:
        zone = await repo.create_zone(db, payload)
        await db.commit()
    except DBAPIError as exc:
        # Raised by PostGIS for malformed GeoJSON or a boundary with no polygonal area.
        await db.rollback()
        raise HTTPException(422, "Invalid boundary geometry.") from exc
    return RiskZoneRead.model_validate(zone)


@router.get(
    "/containing",
    response_model=list[RiskZoneRead],
    summary="Which risk zones contain a point",
)
async def zones_containing(
    db: DbSession,
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
) -> list[RiskZoneRead]:
    return [RiskZoneRead.model_validate(z) for z in await repo.zones_containing(db, lat, lng)]
