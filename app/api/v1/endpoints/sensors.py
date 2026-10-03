from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError

from app.api.deps import DbSession
from app.repositories import sensors as repo
from app.schemas.sensor import SensorReadingCreate, SensorReadingNearby, SensorReadingRead

router = APIRouter(prefix="/sensors", tags=["sensors"])


@router.post(
    "/readings",
    response_model=SensorReadingRead,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a sensor reading",
)
async def ingest_reading(payload: SensorReadingCreate, db: DbSession) -> SensorReadingRead:
    try:
        reading = await repo.create_reading(db, payload)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A reading for this sensor_id at this timestamp already exists.",
        ) from exc
    return SensorReadingRead.model_validate(reading)


@router.get("/latest", response_model=list[SensorReadingRead], summary="Latest reading per sensor")
async def latest_readings(
    db: DbSession, limit: int = Query(500, ge=1, le=5000)
) -> list[SensorReadingRead]:
    rows = await repo.latest_per_sensor(db, limit)
    return [SensorReadingRead.model_validate(r) for r in rows]


@router.get("/nearby", response_model=list[SensorReadingNearby], summary="Readings within a radius")
async def nearby_readings(
    db: DbSession,
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(5000, gt=0, le=200_000),
    since: datetime | None = None,
    limit: int = Query(200, ge=1, le=2000),
) -> list[SensorReadingNearby]:
    rows = await repo.readings_within(db, lat, lng, radius_m, since, limit)
    return [
        SensorReadingNearby.model_validate(
            {**SensorReadingRead.model_validate(r).model_dump(), "distance_m": round(d, 1)}
        )
        for r, d in rows
    ]


@router.get(
    "/{sensor_id}/readings",
    response_model=list[SensorReadingRead],
    summary="Reading history for one sensor",
)
async def sensor_history(
    sensor_id: str,
    db: DbSession,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(500, ge=1, le=10_000),
) -> list[SensorReadingRead]:
    rows = await repo.history_for_sensor(db, sensor_id, since, until, limit)
    return [SensorReadingRead.model_validate(r) for r in rows]
