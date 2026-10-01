from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import BigInteger, CheckConstraint, DateTime, Index, Numeric, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SeverityLevel(enum.StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RiskZone(Base):
    """Assessed risk polygon (e.g. a district). `updated_at` is maintained by
    the `trg_risk_zones_updated_at` trigger, not application code."""

    __tablename__ = "risk_zones"

    zone_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    zone_name: Mapped[str | None] = mapped_column(Text)
    # MultiPolygon accepts single polygons too -- wrap with ST_Multi() on insert.
    boundary: Mapped[WKBElement] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=False), nullable=False
    )
    risk_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    severity_level: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("risk_score BETWEEN 0 AND 100", name="risk_score_range"),
        CheckConstraint("severity_level IN ('Low', 'Medium', 'High')", name="severity_level"),
        # Containment queries only; add a geography index if radius search is needed.
        Index("idx_risk_zones_boundary", "boundary", postgresql_using="gist"),
        Index("idx_risk_zones_severity_level", "severity_level"),
        {"comment": "Assessed landslide risk polygons, colour-coded by severity_level."},
    )
