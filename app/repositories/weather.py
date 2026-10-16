from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RealtimeWeatherLog
from app.schemas.weather import RealtimeWeatherLogCreate


async def insert_logs(db: AsyncSession, logs: Sequence[RealtimeWeatherLogCreate]) -> int:
    """Insert readings, skipping any already stored for (location_name,
    recorded_at). Returns the number of new rows."""
    if not logs:
        return 0
    stmt = (
        insert(RealtimeWeatherLog)
        # `location` is omitted: Postgres generates it from latitude/longitude.
        .values([log.model_dump() for log in logs])
        .on_conflict_do_nothing(constraint="uq_realtime_weather_logs_name_recorded")
        .returning(RealtimeWeatherLog.id)
    )
    inserted = (await db.scalars(stmt)).all()
    await db.flush()
    return len(inserted)
