from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class RealtimeWeatherLogCreate(BaseModel):
    """One normalised weather reading, ready for the repository. Bounds mirror
    the CHECK constraints on realtime_weather_logs."""

    location_name: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    rainfall_mm: Decimal = Field(..., ge=0, le=Decimal("99999.99"))
    humidity: Decimal = Field(..., ge=0, le=100)
    temperature: Decimal = Field(..., ge=-60, le=60)
    recorded_at: datetime
