"""Initial PostGIS schema: sensors_data, hazard_reports, risk_zones.

Faithful port of legacy/001_landslide_ews_schema.sql (see that file for the
EXPLAIN ANALYZE rationale behind the geography expression indexes). The only
difference is that CHECK constraints get explicit names matching the ORM
naming convention, so autogenerate diffs stay clean.

Written as raw SQL rather than op.create_table because the PostGIS-specific
pieces (expression indexes, trigger) read far more clearly this way.

Revision ID: 0001
Revises:
Create Date: 2026-10-01
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


UPGRADE = [
    # gen_random_uuid() is core since PG13; pgcrypto covers older versions.
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
    "CREATE EXTENSION IF NOT EXISTS postgis",
    """
    CREATE OR REPLACE FUNCTION set_updated_at()
    RETURNS trigger AS $$
    BEGIN
      NEW.updated_at := now();
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql
    """,
    # --- sensors_data -------------------------------------------------------
    """
    CREATE TABLE sensors_data (
        id              BIGSERIAL PRIMARY KEY,
        sensor_id       TEXT NOT NULL,
        location        geometry(Point, 4326) NOT NULL,
        rainfall_mm     NUMERIC(7, 2) NOT NULL,
        soil_moisture   NUMERIC(5, 2) NOT NULL,
        "timestamp"     TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_sensors_data_sensor_timestamp UNIQUE (sensor_id, "timestamp"),
        CONSTRAINT chk_sensors_data_rainfall_nonneg CHECK (rainfall_mm >= 0),
        CONSTRAINT chk_sensors_data_soil_moisture_pct CHECK (soil_moisture BETWEEN 0 AND 100)
    )
    """,
    "COMMENT ON TABLE sensors_data IS "
    "'Time-series rainfall/soil-moisture readings from field sensors.'",
    "CREATE INDEX idx_sensors_data_location ON sensors_data USING GIST (location)",
    "CREATE INDEX idx_sensors_data_location_geog "
    "ON sensors_data USING GIST ((location::geography))",
    'CREATE INDEX idx_sensors_data_sensor_timestamp ON sensors_data (sensor_id, "timestamp" DESC)',
    # --- hazard_reports -----------------------------------------------------
    """
    CREATE TABLE hazard_reports (
        report_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        location        geometry(Point, 4326) NOT NULL,
        hazard_type     TEXT NOT NULL,
        photo_url       TEXT,
        status          TEXT NOT NULL DEFAULT 'pending',
        reported_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        synced_at       TIMESTAMPTZ,
        CONSTRAINT chk_hazard_reports_hazard_type
            CHECK (hazard_type IN ('road_crack', 'active_mudslide', 'blocked_road')),
        CONSTRAINT chk_hazard_reports_status
            CHECK (status IN ('pending', 'verified', 'resolved', 'rejected'))
    )
    """,
    "COMMENT ON TABLE hazard_reports IS "
    "'Crowdsourced hazard reports submitted from the field PWA.'",
    "CREATE INDEX idx_hazard_reports_location ON hazard_reports USING GIST (location)",
    "CREATE INDEX idx_hazard_reports_location_geog "
    "ON hazard_reports USING GIST ((location::geography))",
    "CREATE INDEX idx_hazard_reports_status ON hazard_reports (status)",
    "CREATE INDEX idx_hazard_reports_hazard_type ON hazard_reports (hazard_type)",
    # --- risk_zones ---------------------------------------------------------
    """
    CREATE TABLE risk_zones (
        zone_id         BIGSERIAL PRIMARY KEY,
        zone_name       TEXT,
        boundary        geometry(MultiPolygon, 4326) NOT NULL,
        risk_score      NUMERIC(5, 2) NOT NULL,
        severity_level  TEXT NOT NULL,
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT chk_risk_zones_risk_score_range CHECK (risk_score BETWEEN 0 AND 100),
        CONSTRAINT chk_risk_zones_severity_level
            CHECK (severity_level IN ('Low', 'Medium', 'High'))
    )
    """,
    "COMMENT ON TABLE risk_zones IS "
    "'Assessed landslide risk polygons, colour-coded by severity_level.'",
    "CREATE INDEX idx_risk_zones_boundary ON risk_zones USING GIST (boundary)",
    "CREATE INDEX idx_risk_zones_severity_level ON risk_zones (severity_level)",
    """
    CREATE TRIGGER trg_risk_zones_updated_at
        BEFORE UPDATE ON risk_zones
        FOR EACH ROW
        EXECUTE FUNCTION set_updated_at()
    """,
]

DOWNGRADE = [
    "DROP TABLE IF EXISTS risk_zones",
    "DROP TABLE IF EXISTS hazard_reports",
    "DROP TABLE IF EXISTS sensors_data",
    "DROP FUNCTION IF EXISTS set_updated_at()",
    # Extensions are left installed: other schemas may depend on them.
]


def upgrade() -> None:
    for stmt in UPGRADE:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in DOWNGRADE:
        op.execute(stmt)
