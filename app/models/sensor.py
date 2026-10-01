from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    Numeric,
    Text,
    UniqueConstraint,
    cast,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import PlainGeography


class SensorReading(Base):
    """One rainfall/soil-moisture reading. `sensor_id` is the physical device
    and repeats across rows; `id` is the row's surrogate key."""

    __tablename__ = "sensors_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[WKBElement] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=False), nullable=False
    )
    rainfall_mm: Mapped[Decimal] = mapped_column(Numeric(7, 2), nullable=False)
    soil_moisture: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("sensor_id", "timestamp", name="uq_sensors_data_sensor_timestamp"),
        CheckConstraint("rainfall_mm >= 0", name="rainfall_nonneg"),
        CheckConstraint("soil_moisture BETWEEN 0 AND 100", name="soil_moisture_pct"),
        # Point-in-polygon / bbox queries.
        Index("idx_sensors_data_location", "location", postgresql_using="gist"),
        {"comment": "Time-series rainfall/soil-moisture readings from field sensors."},
    )


# Expression indexes are declared after the class so they can reference the
# mapped attributes.
#
# Metres-based radius queries cast to geography; a plain geometry GIST index is
# NOT used for those (see legacy/001_landslide_ews_schema.sql). Queries must use
# exactly `cast(SensorReading.location, PlainGeography())` to hit this index.
Index(
    "idx_sensors_data_location_geog",
    cast(SensorReading.location, PlainGeography()),
    postgresql_using="gist",
)
# "Latest reading per sensor" / time-range queries.
Index(
    "idx_sensors_data_sensor_timestamp",
    SensorReading.sensor_id,
    SensorReading.timestamp.desc(),
)
