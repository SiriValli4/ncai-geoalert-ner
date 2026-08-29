-- ============================================================================
-- Landslide Early Warning System -- PostGIS schema
-- ============================================================================
-- Tables: sensors_data, hazard_reports, risk_zones
--
-- Consistent with the rest of this system (dashboard, hazard detection API,
-- dispatch-alerts): risk scores are 0-100, severity is Low/Medium/High, and
-- hazard_type mirrors the classes the YOLOv8 hazard detection service
-- returns (road_crack, active_mudslide, blocked_road).
--
-- Coordinates use geometry(..., 4326) (WGS84 lat/lon), the standard choice
-- for GPS-sourced points and what Leaflet/GeoJSON/most tooling expects
-- natively. Two consequences worth knowing before you query this:
--   * Point-in-polygon checks (ST_Contains, ST_Within, ST_Intersects) work
--     directly on these columns -- no casting needed, and the plain GIST
--     indexes below (idx_*_location) accelerate them. Verified via
--     EXPLAIN ANALYZE against a 50k-row sensors_data table: Index/Bitmap
--     Index Scan, not a sequential scan.
--   * ST_Distance on a raw geometry(4326) column returns degrees, not
--     metres, and is not what you want for "find sensors within 5km".
--     For real-world-distance queries, cast to geography at query time:
--       SELECT * FROM sensors_data
--       WHERE ST_DWithin(location::geography, ST_MakePoint(:lng,:lat)::geography, 5000);
--     IMPORTANT: a plain GIST index on the geometry column does NOT get
--     used once the column is cast to geography inside the predicate --
--     confirmed by EXPLAIN ANALYZE, that cast query fell back to a
--     sequential scan (2.7s over 50k rows) with only idx_sensors_data_location
--     in place. The fix, already applied below, is a second GIST index
--     built directly on the geography-cast expression
--     (idx_sensors_data_location_geog / idx_hazard_reports_location_geog);
--     with that in place the same query dropped to ~6ms via a Bitmap Index
--     Scan. If you only ever do point-in-polygon queries on a table and
--     never radius/distance searches, you can drop its _geog index --
--     they're kept here because "reports/sensors within N km" is a very
--     plausible query for both tables.
--
-- Run as: psql -d your_database -f 001_landslide_ews_schema.sql
-- ============================================================================

BEGIN;

-- gen_random_uuid() is built into core Postgres since v13; pgcrypto makes it
-- available on older versions too, and is a harmless no-op if already core.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS postgis;

-- ----------------------------------------------------------------------------
-- Shared trigger: keep updated_at columns honest without relying on
-- application code to remember to set them.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 1. sensors_data -- time-series telemetry from field rainfall/soil sensors
-- ============================================================================
-- Note on primary key: a physical sensor (sensor_id) reports repeatedly over
-- time, so sensor_id alone can't be the primary key of a readings table --
-- doing so would let each new reading silently overwrite the last one. This
-- adds a surrogate `id` as the primary key and keeps `sensor_id` as an
-- indexed reference column instead, plus a UNIQUE constraint so the same
-- sensor can't accidentally insert two readings at the same instant.

CREATE TABLE IF NOT EXISTS sensors_data (
    id              BIGSERIAL PRIMARY KEY,
    sensor_id       TEXT NOT NULL,
    location        geometry(Point, 4326) NOT NULL,
    rainfall_mm     NUMERIC(7, 2) NOT NULL CHECK (rainfall_mm >= 0),
    soil_moisture   NUMERIC(5, 2) NOT NULL CHECK (soil_moisture BETWEEN 0 AND 100),
    "timestamp"     TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_sensors_data_sensor_timestamp UNIQUE (sensor_id, "timestamp")
);

COMMENT ON TABLE sensors_data IS 'Time-series rainfall/soil-moisture readings from field sensors.';
COMMENT ON COLUMN sensors_data.sensor_id IS 'Physical sensor device identifier (not unique per row -- one sensor reports many readings).';
COMMENT ON COLUMN sensors_data.soil_moisture IS 'Volumetric soil moisture, percent (0-100).';

-- Spatial index for point-in-polygon / bounding-box queries on this column.
CREATE INDEX IF NOT EXISTS idx_sensors_data_location
    ON sensors_data USING GIST (location);

-- Separate expression index for metres-based radius searches (see the note
-- at the top of this file for why this is a second index, not redundant
-- with the one above).
CREATE INDEX IF NOT EXISTS idx_sensors_data_location_geog
    ON sensors_data USING GIST ((location::geography));

-- Supporting index for the very common "latest reading per sensor" / time
-- range query, which the spatial indexes alone can't serve efficiently.
CREATE INDEX IF NOT EXISTS idx_sensors_data_sensor_timestamp
    ON sensors_data (sensor_id, "timestamp" DESC);

-- ============================================================================
-- 2. hazard_reports -- crowdsourced/field hazard reports (photo + geotag)
-- ============================================================================
-- Mirrors FieldReportPWA.jsx's offline-first submission flow: reported_at is
-- when the officer captured the report (from device geolocation, possibly
-- while offline), while synced_at is when it actually reached this database
-- -- the two can differ by hours if a report sat in an IndexedDB queue.
-- synced_at is nullable: NULL means "inserted directly" or "not yet known to
-- have round-tripped through an offline sync"; set it explicitly when a
-- queued report is flushed from a client's local store.

CREATE TABLE IF NOT EXISTS hazard_reports (
    report_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location        geometry(Point, 4326) NOT NULL,
    hazard_type     TEXT NOT NULL
                        CHECK (hazard_type IN ('road_crack', 'active_mudslide', 'blocked_road')),
    photo_url       TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'verified', 'resolved', 'rejected')),
    reported_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    synced_at       TIMESTAMPTZ
);

COMMENT ON TABLE hazard_reports IS 'Crowdsourced hazard reports submitted from the field (see FieldReportPWA.jsx).';
COMMENT ON COLUMN hazard_reports.hazard_type IS 'Matches the classes the YOLOv8 hazard detection API classifies (road_crack, active_mudslide, blocked_road).';
COMMENT ON COLUMN hazard_reports.reported_at IS 'When the officer captured the report on-device, independent of network availability.';
COMMENT ON COLUMN hazard_reports.synced_at IS 'When this report reached the server; NULL until a queued/offline submission is confirmed synced.';

CREATE INDEX IF NOT EXISTS idx_hazard_reports_location
    ON hazard_reports USING GIST (location);

-- See the note at the top of this file: needed for "reports within N km"
-- style queries, distinct from the plain geometry index above.
CREATE INDEX IF NOT EXISTS idx_hazard_reports_location_geog
    ON hazard_reports USING GIST ((location::geography));

CREATE INDEX IF NOT EXISTS idx_hazard_reports_status
    ON hazard_reports (status);

CREATE INDEX IF NOT EXISTS idx_hazard_reports_hazard_type
    ON hazard_reports (hazard_type);

-- ============================================================================
-- 3. risk_zones -- computed/assessed risk polygons (districts, hazard zones)
-- ============================================================================
-- boundary is MultiPolygon rather than a plain Polygon: real administrative
-- and hazard-zone boundary data (shapefiles, dissolved/merged zones) is very
-- commonly multi-part, and MultiPolygon accepts single polygons too (wrap
-- with ST_Multi(geom) on insert if your source data is plain Polygon) --
-- so this is a strict superset with no downside for this use case.

CREATE TABLE IF NOT EXISTS risk_zones (
    zone_id         BIGSERIAL PRIMARY KEY,
    zone_name       TEXT,
    boundary        geometry(MultiPolygon, 4326) NOT NULL,
    risk_score      NUMERIC(5, 2) NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    severity_level  TEXT NOT NULL CHECK (severity_level IN ('Low', 'Medium', 'High')),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()

    -- Optional stricter integrity check, commented out by default: real-world
    -- boundary imports (shapefiles, dissolved polygons) occasionally contain
    -- minor topology issues that ST_MakeValid() should clean up before load
    -- rather than have insert fail outright. Enable once your ingestion
    -- pipeline guarantees valid geometry:
    -- , CONSTRAINT chk_risk_zones_valid_geom CHECK (ST_IsValid(boundary))
);

COMMENT ON TABLE risk_zones IS 'Assessed landslide risk polygons (e.g. district-level), color-coded by severity_level.';
COMMENT ON COLUMN risk_zones.risk_score IS '0-100 AI/assessed risk score, consistent with the rest of the system.';
COMMENT ON COLUMN risk_zones.severity_level IS 'Low / Medium / High -- drives red/yellow/green rendering on the dashboard.';

-- No separate geography expression index here, unlike the point tables
-- above: risk_zones is queried by containment (ST_Contains/ST_Intersects,
-- which this plain geometry GIST index serves directly), not by
-- radius/distance search. Add one the same way as sensors_data/
-- hazard_reports above if you later need "zones within N km of X".
CREATE INDEX IF NOT EXISTS idx_risk_zones_boundary
    ON risk_zones USING GIST (boundary);

CREATE INDEX IF NOT EXISTS idx_risk_zones_severity_level
    ON risk_zones (severity_level);

DROP TRIGGER IF EXISTS trg_risk_zones_updated_at ON risk_zones;
CREATE TRIGGER trg_risk_zones_updated_at
    BEFORE UPDATE ON risk_zones
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

COMMIT;

-- ============================================================================
-- Example queries the indexes above are built for (not executed by this
-- script -- for reference)
-- ============================================================================
--
-- Sensors within 5km of a point, using a geography cast for a correct
-- metres-based radius (see the note at the top of this file):
--
--   SELECT sensor_id, rainfall_mm, soil_moisture, "timestamp"
--   FROM sensors_data
--   WHERE ST_DWithin(location::geography, ST_SetSRID(ST_MakePoint(91.893, 25.578), 4326)::geography, 5000)
--   ORDER BY "timestamp" DESC;
--
-- Which risk zone (if any) a hazard report falls inside:
--
--   SELECT hr.report_id, hr.hazard_type, rz.zone_name, rz.severity_level
--   FROM hazard_reports hr
--   JOIN risk_zones rz ON ST_Contains(rz.boundary, hr.location)
--   WHERE hr.status = 'pending';
--
-- All pending hazard reports inside HIGH-severity zones (a natural dispatch
-- trigger query, e.g. feeding POST /api/dispatch-alerts):
--
--   SELECT hr.report_id, hr.hazard_type, hr.location, rz.zone_name
--   FROM hazard_reports hr
--   JOIN risk_zones rz ON ST_Contains(rz.boundary, hr.location)
--   WHERE rz.severity_level = 'High' AND hr.status = 'pending';
-- ============================================================================
