"""Import every model here so `Base.metadata` is complete for Alembic."""

from app.models.hazard_report import HazardReport, HazardType, ReportStatus
from app.models.risk_zone import RiskZone, SeverityLevel
from app.models.sensor import SensorReading

__all__ = [
    "HazardReport",
    "HazardType",
    "ReportStatus",
    "RiskZone",
    "SensorReading",
    "SeverityLevel",
]
