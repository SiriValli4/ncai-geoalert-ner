"""Import every model here so `Base.metadata` is complete for Alembic."""

from app.models.hazard_report import HazardReport, HazardType, ReportStatus
from app.models.risk_zone import RiskZone, SeverityLevel
from app.models.sensor import SensorReading
from app.models.weather import RealtimeWeatherLog

__all__ = [
    "HazardReport",
    "RealtimeWeatherLog",
    "HazardType",
    "ReportStatus",
    "RiskZone",
    "SensorReading",
    "SeverityLevel",
]
