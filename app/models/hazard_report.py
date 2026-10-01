from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import CheckConstraint, DateTime, Index, Numeric, Text, Uuid, cast, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import PlainGeography


class HazardType(enum.StrEnum):
    # The first three match the YOLOv8 detector's classes. `rockfall` is
    # human-reported only (the field PWA offers it; the detector can't classify it).
    ROAD_CRACK = "road_crack"
    ACTIVE_MUDSLIDE = "active_mudslide"
    BLOCKED_ROAD = "blocked_road"
    ROCKFALL = "rockfall"


class ReportStatus(enum.StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    RESOLVED = "resolved"
    REJECTED = "rejected"


def _in_list(column: str, enum_cls: type[enum.Enum]) -> str:
    values = ", ".join(f"'{m.value}'" for m in enum_cls)
    return f"{column} IN ({values})"


class HazardReport(Base):
    """A field report. `reported_at` is capture time on-device (possibly while
    offline); `synced_at` is when it reached the server."""

    __tablename__ = "hazard_reports"

    report_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    location: Mapped[WKBElement] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=False), nullable=False
    )
    hazard_type: Mapped[str] = mapped_column(Text, nullable=False)
    photo_url: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=ReportStatus.PENDING.value
    )
    notes: Mapped[str | None] = mapped_column(Text)
    location_accuracy_m: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(_in_list("hazard_type", HazardType), name="hazard_type"),
        CheckConstraint(_in_list("status", ReportStatus), name="status"),
        Index("idx_hazard_reports_location", "location", postgresql_using="gist"),
        Index("idx_hazard_reports_status", "status"),
        Index("idx_hazard_reports_hazard_type", "hazard_type"),
        {"comment": "Crowdsourced hazard reports submitted from the field PWA."},
    )


# Declared after the class so it can reference the mapped attribute. Radius
# queries must use exactly `cast(HazardReport.location, PlainGeography())`.
Index(
    "idx_hazard_reports_location_geog",
    cast(HazardReport.location, PlainGeography()),
    postgresql_using="gist",
)
