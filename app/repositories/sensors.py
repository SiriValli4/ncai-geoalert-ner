from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SensorReading
from app.repositories.spatial import as_geography, make_point
from app.schemas.geo import point_to_wkb
from app.schemas.sensor import SensorReadingCreate


async def create_reading(db: AsyncSession, data: SensorReadingCreate) -> SensorReading:
    reading = SensorReading(
        sensor_id=data.sensor_id,
        location=point_to_wkb(data.latitude, data.longitude),
        rainfall_mm=data.rainfall_mm,
        soil_moisture=data.soil_moisture,
    )
    if data.timestamp is not None:
        reading.timestamp = data.timestamp
    db.add(reading)
    await db.flush()
    await db.refresh(reading)
    return reading


async def latest_per_sensor(db: AsyncSession, limit: int = 500) -> list[SensorReading]:
    """Most recent reading for each sensor (served by idx_sensors_data_sensor_timestamp)."""
    stmt = (
        select(SensorReading)
        .distinct(SensorReading.sensor_id)
        .order_by(SensorReading.sensor_id, SensorReading.timestamp.desc())
        .limit(limit)
    )
    return list((await db.scalars(stmt)).all())


async def history_for_sensor(
    db: AsyncSession,
    sensor_id: str,
    since: datetime | None,
    until: datetime | None,
    limit: int,
) -> list[SensorReading]:
    stmt = select(SensorReading).where(SensorReading.sensor_id == sensor_id)
    if since is not None:
        stmt = stmt.where(SensorReading.timestamp >= since)
    if until is not None:
        stmt = stmt.where(SensorReading.timestamp < until)
    stmt = stmt.order_by(SensorReading.timestamp.desc()).limit(limit)
    return list((await db.scalars(stmt)).all())


async def readings_within(
    db: AsyncSession,
    latitude: float,
    longitude: float,
    radius_m: float,
    since: datetime | None,
    limit: int,
) -> list[tuple[SensorReading, float]]:
    """Readings within `radius_m` metres, nearest first, with distance in metres."""
    origin = as_geography(make_point(latitude, longitude))
    loc = as_geography(SensorReading.location)
    distance = func.ST_Distance(loc, origin).label("distance_m")

    stmt = select(SensorReading, distance).where(func.ST_DWithin(loc, origin, radius_m))
    if since is not None:
        stmt = stmt.where(SensorReading.timestamp >= since)
    stmt = stmt.order_by(distance, SensorReading.timestamp.desc()).limit(limit)

    rows = (await db.execute(stmt)).all()
    return [(row[0], float(row[1])) for row in rows]
