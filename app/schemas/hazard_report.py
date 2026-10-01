from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.hazard_report import HazardType, ReportStatus
from app.schemas.geo import PointFromDb

# The field PWA (frontend/src/FieldReportPWA.jsx) sends human-readable labels.
# Canonical values are accepted too, so API clients can send either.
PWA_LABEL_TO_HAZARD_TYPE: dict[str, HazardType] = {
    "surface crack": HazardType.ROAD_CRACK,
    "rockfall": HazardType.ROCKFALL,
    "blocked road": HazardType.BLOCKED_ROAD,
    "mudslide": HazardType.ACTIVE_MUDSLIDE,
}


def normalize_hazard_type(raw: str) -> HazardType:
    """Map a PWA label or canonical value to a HazardType. Raises ValueError."""
    key = raw.strip().lower()
    if key in PWA_LABEL_TO_HAZARD_TYPE:
        return PWA_LABEL_TO_HAZARD_TYPE[key]
    return HazardType(key)


class HazardReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    report_id: uuid.UUID
    location: PointFromDb
    hazard_type: HazardType
    photo_url: str | None
    status: ReportStatus
    notes: str | None
    location_accuracy_m: float | None
    reported_at: datetime
    synced_at: datetime | None


class HazardReportStatusUpdate(BaseModel):
    status: ReportStatus


class HazardReportInZone(BaseModel):
    """A pending report inside a High-severity zone -- dispatch candidates."""

    report: HazardReportRead
    zone_id: int
    zone_name: str | None
