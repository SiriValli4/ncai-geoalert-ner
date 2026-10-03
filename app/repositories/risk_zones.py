from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RiskZone, SeverityLevel
from app.repositories.spatial import SRID, make_point
from app.schemas.risk_zone import RiskZoneCreate


class InvalidBoundaryError(ValueError):
    """The boundary has no polygonal area, even after repair."""


async def create_zone(db: AsyncSession, data: RiskZoneCreate) -> RiskZone:
    # ST_Multi promotes a Polygon to MultiPolygon; ST_MakeValid repairs the
    # minor topology issues common in shapefile-derived boundaries.
    geom = func.ST_Multi(
        func.ST_CollectionExtract(
            func.ST_MakeValid(
                func.ST_SetSRID(func.ST_GeomFromGeoJSON(json.dumps(data.boundary)), SRID)
            ),
            3,  # keep polygonal parts only
        )
    )
    # The repair chain never fails on junk input -- a degenerate ring (e.g. two
    # points) comes out as an EMPTY MultiPolygon, which would otherwise be
    # stored as an invisible zero-area risk zone. Reject it explicitly.
    if await db.scalar(select(func.ST_IsEmpty(geom))):
        raise InvalidBoundaryError("boundary has no polygonal area")

    zone = RiskZone(
        zone_name=data.zone_name,
        boundary=geom,
        risk_score=data.risk_score,
        severity_level=data.severity_level.value,
    )
    db.add(zone)
    await db.flush()
    await db.refresh(zone)
    return zone


async def list_zones(
    db: AsyncSession, severity: SeverityLevel | None, limit: int
) -> list[RiskZone]:
    stmt = select(RiskZone)
    if severity is not None:
        stmt = stmt.where(RiskZone.severity_level == severity.value)
    stmt = stmt.order_by(RiskZone.risk_score.desc()).limit(limit)
    return list((await db.scalars(stmt)).all())


async def zones_containing(db: AsyncSession, latitude: float, longitude: float) -> list[RiskZone]:
    stmt = (
        select(RiskZone)
        .where(func.ST_Contains(RiskZone.boundary, make_point(latitude, longitude)))
        .order_by(RiskZone.risk_score.desc())
    )
    return list((await db.scalars(stmt)).all())
