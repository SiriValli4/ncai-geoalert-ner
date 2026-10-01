from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.risk_zone import SeverityLevel
from app.schemas.geo import GeoJSONFromDb


class RiskZoneCreate(BaseModel):
    zone_name: str | None = Field(None, max_length=200)
    boundary: dict[str, Any] = Field(
        ..., description="GeoJSON Polygon or MultiPolygon in WGS84 (EPSG:4326)."
    )
    risk_score: float = Field(..., ge=0, le=100)
    severity_level: SeverityLevel

    @field_validator("boundary")
    @classmethod
    def _polygonal(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("boundary must be a GeoJSON Polygon or MultiPolygon")
        if not v.get("coordinates"):
            raise ValueError("boundary.coordinates is required")
        return v


class RiskZoneRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    zone_id: int
    zone_name: str | None
    boundary: GeoJSONFromDb
    risk_score: float
    severity_level: SeverityLevel
    updated_at: datetime


class RiskZoneFeatureCollection(BaseModel):
    """GeoJSON FeatureCollection -- drops straight into Leaflet's <GeoJSON>."""

    type: str = "FeatureCollection"
    features: list[dict[str, Any]]

    @classmethod
    def from_zones(cls, zones: list[RiskZoneRead]) -> RiskZoneFeatureCollection:
        return cls(
            features=[
                {
                    "type": "Feature",
                    "id": z.zone_id,
                    "geometry": z.boundary,
                    "properties": {
                        "zone_id": z.zone_id,
                        "zone_name": z.zone_name,
                        "risk_score": z.risk_score,
                        "severity_level": z.severity_level.value,
                        "updated_at": z.updated_at.isoformat(),
                    },
                }
                for z in zones
            ]
        )
