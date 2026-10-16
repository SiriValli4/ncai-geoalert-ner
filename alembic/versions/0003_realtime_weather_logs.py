"""Add realtime_weather_logs (rainfall, humidity, temperature by place).

Written by the OpenWeatherMap ingestion job (app/services/weather_ingestion.py).

`location` is a STORED generated column built from latitude/longitude, so
writers only supply the two numbers and the geometry can never disagree with
them. It carries the usual GIST + `(location::geography)` indexes.

Raw SQL rather than the autogenerate draft, like 0001: the draft lost the table
comment and tried to drop the expression index as if it were a column index.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-16
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


UPGRADE = [
    """
    CREATE TABLE realtime_weather_logs (
        id              BIGSERIAL,
        location_name   TEXT NOT NULL,
        latitude        DOUBLE PRECISION NOT NULL,
        longitude       DOUBLE PRECISION NOT NULL,
        location        geometry(Point, 4326) NOT NULL
            GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED,
        rainfall_mm     NUMERIC(7, 2) NOT NULL,
        humidity        NUMERIC(5, 2) NOT NULL,
        temperature     NUMERIC(5, 2) NOT NULL,
        recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT pk_realtime_weather_logs PRIMARY KEY (id),
        CONSTRAINT uq_realtime_weather_logs_name_recorded UNIQUE (location_name, recorded_at),
        CONSTRAINT chk_realtime_weather_logs_latitude_range CHECK (latitude BETWEEN -90 AND 90),
        CONSTRAINT chk_realtime_weather_logs_longitude_range
            CHECK (longitude BETWEEN -180 AND 180),
        CONSTRAINT chk_realtime_weather_logs_rainfall_nonneg CHECK (rainfall_mm >= 0),
        CONSTRAINT chk_realtime_weather_logs_humidity_pct CHECK (humidity BETWEEN 0 AND 100),
        CONSTRAINT chk_realtime_weather_logs_temperature_celsius
            CHECK (temperature BETWEEN -60 AND 60)
    )
    """,
    "COMMENT ON TABLE realtime_weather_logs IS "
    "'Weather readings by place (rainfall, humidity, temperature), from OpenWeatherMap.'",
    "CREATE INDEX idx_realtime_weather_logs_location "
    "ON realtime_weather_logs USING GIST (location)",
    "CREATE INDEX idx_realtime_weather_logs_location_geog "
    "ON realtime_weather_logs USING GIST ((location::geography))",
]

DOWNGRADE = [
    "DROP TABLE IF EXISTS realtime_weather_logs",
]


def upgrade() -> None:
    for stmt in UPGRADE:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in DOWNGRADE:
        op.execute(stmt)
