from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.geo import PointFromDb


class SensorReadingCreate(BaseModel):
    sensor_id: str = Field(..., min_length=1, max_length=128)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    rainfall_mm: float = Field(..., ge=0, le=99999.99)
    soil_moisture: float = Field(..., ge=0, le=100, description="Volumetric soil moisture, %.")
    timestamp: datetime | None = Field(None, description="Defaults to server time if omitted.")


class SensorReadingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sensor_id: str
    location: PointFromDb
    rainfall_mm: float
    soil_moisture: float
    timestamp: datetime


class SensorReadingNearby(SensorReadingRead):
    distance_m: float
