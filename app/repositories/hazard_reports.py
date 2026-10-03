from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import HazardReport, HazardType, ReportStatus, RiskZone, SeverityLevel
from app.repositories.spatial import as_geography, make_point
from app.schemas.geo import point_to_wkb


async def create_report(
    db: AsyncSession,
    *,
    hazard_type: HazardType,
    latitude: float,
    longitude: float,
    reported_at: datetime | None,
    notes: str | None,
    accuracy_m: float | None,
    photo_url: str | None,
) -> HazardReport:
    report = HazardReport(
        hazard_type=hazard_type.value,
        location=point_to_wkb(latitude, longitude),
        notes=notes,
        location_accuracy_m=accuracy_m,
        photo_url=photo_url,
        # Arrival time at the server; differs from reported_at for
        # reports that sat in the PWA's offline queue.
        synced_at=datetime.now(UTC),
    )
    if reported_at is not None:
        report.reported_at = reported_at
    db.add(report)
    await db.flush()
    await db.refresh(report)
    return report


async def get_report(db: AsyncSession, report_id: uuid.UUID) -> HazardReport | None:
    return await db.get(HazardReport, report_id)


async def list_reports(
    db: AsyncSession,
    *,
    status: ReportStatus | None,
    hazard_type: HazardType | None,
    near: tuple[float, float] | None,
    radius_m: float | None,
    limit: int,
    offset: int,
) -> list[HazardReport]:
    stmt = select(HazardReport)
    if status is not None:
        stmt = stmt.where(HazardReport.status == status.value)
    if hazard_type is not None:
        stmt = stmt.where(HazardReport.hazard_type == hazard_type.value)
    if near is not None and radius_m is not None:
        origin = as_geography(make_point(*near))
        stmt = stmt.where(func.ST_DWithin(as_geography(HazardReport.location), origin, radius_m))
    stmt = stmt.order_by(HazardReport.reported_at.desc()).limit(limit).offset(offset)
    return list((await db.scalars(stmt)).all())


async def update_status(db: AsyncSession, report: HazardReport, status: ReportStatus) -> None:
    report.status = status.value
    await db.flush()


async def pending_in_high_risk_zones(
    db: AsyncSession, limit: int
) -> list[tuple[HazardReport, int, str | None]]:
    """Pending reports inside High-severity zones -- the natural dispatch trigger."""
    stmt = (
        select(HazardReport, RiskZone.zone_id, RiskZone.zone_name)
        .join(RiskZone, func.ST_Contains(RiskZone.boundary, HazardReport.location))
        .where(
            RiskZone.severity_level == SeverityLevel.HIGH.value,
            HazardReport.status == ReportStatus.PENDING.value,
        )
        .order_by(HazardReport.reported_at.desc())
        .limit(limit)
    )
    return [(r[0], r[1], r[2]) for r in (await db.execute(stmt)).all()]
