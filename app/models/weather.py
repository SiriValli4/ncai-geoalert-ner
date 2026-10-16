from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    Double,
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


class RealtimeWeatherLog(Base):
    """One weather reading (rainfall, humidity, temperature) at a named place,
    written hourly by the OpenWeatherMap job (services/weather_ingestion.py).

    `latitude`/`longitude` are what callers write. `location` is generated from
    them by Postgres, so the two can never disagree, and it exists only so
    spatial queries get a real geometry column with GIST indexes.
    """

    __tablename__ = "realtime_weather_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    location_name: Mapped[str] = mapped_column(Text, nullable=False)
    latitude: Mapped[float] = mapped_column(Double, nullable=False)
    longitude: Mapped[float] = mapped_column(Double, nullable=False)
    # x=longitude, y=latitude (see CLAUDE.md, PostGIS conventions).
    location: Mapped[WKBElement] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=False),
        Computed("ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)", persisted=True),
    )
    # Rain over the hour before recorded_at (OWM `rain.1h`).
    rainfall_mm: Mapped[Decimal] = mapped_column(Numeric(7, 2), nullable=False)
    # Relative humidity, percent.
    humidity: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    # Air temperature, degrees Celsius.
    temperature: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    # The provider's observation time (OWM `dt`), not when we fetched it.
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # One reading per place per instant. Makes ingestion idempotent (re-fetching
        # an unchanged OWM observation adds nothing) and serves "latest for a place".
        UniqueConstraint(
            "location_name", "recorded_at", name="uq_realtime_weather_logs_name_recorded"
        ),
        CheckConstraint("latitude BETWEEN -90 AND 90", name="latitude_range"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="longitude_range"),
        CheckConstraint("rainfall_mm >= 0", name="rainfall_nonneg"),
        CheckConstraint("humidity BETWEEN 0 AND 100", name="humidity_pct"),
        # Generous physical bounds: catches unit mix-ups (Kelvin, Fahrenheit).
        CheckConstraint("temperature BETWEEN -60 AND 60", name="temperature_celsius"),
        Index("idx_realtime_weather_logs_location", "location", postgresql_using="gist"),
        {
            "comment": "Weather readings by place (rainfall, humidity, temperature), "
            "from OpenWeatherMap."
        },
    )


# Metres-based radius queries; see the note in models/sensor.py.
Index(
    "idx_realtime_weather_logs_location_geog",
    cast(RealtimeWeatherLog.location, PlainGeography()),
    postgresql_using="gist",
)
