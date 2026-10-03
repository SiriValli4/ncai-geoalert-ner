"""Geometry conversion and SQL-shape tests -- no database required."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql

from app.models import HazardReport, HazardType, SensorReading
from app.repositories.spatial import as_geography, make_point
from app.schemas.geo import GeoPoint, _wkb_to_point, point_to_wkb
from app.schemas.hazard_report import normalize_hazard_type


def _sql(stmt) -> str:  # type: ignore[no-untyped-def]
    return str(stmt.compile(dialect=postgresql.dialect())).replace("\n", " ")


def test_point_roundtrip_preserves_axis_order() -> None:
    wkb = point_to_wkb(latitude=25.578, longitude=91.893)
    assert wkb.srid == 4326
    pt = GeoPoint.model_validate(_wkb_to_point(wkb))
    assert pt.latitude == pytest.approx(25.578)
    assert pt.longitude == pytest.approx(91.893)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Surface Crack", HazardType.ROAD_CRACK),
        ("Rockfall", HazardType.ROCKFALL),
        ("Blocked Road", HazardType.BLOCKED_ROAD),
        ("active_mudslide", HazardType.ACTIVE_MUDSLIDE),
        ("  ROAD_CRACK ", HazardType.ROAD_CRACK),
    ],
)
def test_pwa_labels_map_to_hazard_types(raw: str, expected: HazardType) -> None:
    assert normalize_hazard_type(raw) is expected


def test_unknown_hazard_type_raises() -> None:
    with pytest.raises(ValueError):
        normalize_hazard_type("volcano")


def test_radius_query_uses_plain_geography_cast() -> None:
    """The geography expression indexes are built on `location::geography`.
    A cast with a typmod (e.g. geography(GEOMETRY,-1)) would not match them
    and silently fall back to a sequential scan."""
    origin = as_geography(make_point(25.5, 91.8))
    stmt = select(SensorReading).where(
        func.ST_DWithin(as_geography(SensorReading.location), origin, 5000)
    )
    sql = _sql(stmt)
    assert "CAST(sensors_data.location AS geography)" in sql
    assert "ST_MakePoint" in sql


def test_model_indexes_match_migration_names() -> None:
    sensor_idx = {i.name for i in SensorReading.__table__.indexes}
    report_idx = {i.name for i in HazardReport.__table__.indexes}
    assert {
        "idx_sensors_data_location",
        "idx_sensors_data_location_geog",
        "idx_sensors_data_sensor_timestamp",
    } <= sensor_idx
    assert {"idx_hazard_reports_location", "idx_hazard_reports_location_geog"} <= report_idx


def test_check_constraint_names_follow_convention() -> None:
    names = {c.name for c in HazardReport.__table__.constraints if c.name}
    assert "chk_hazard_reports_hazard_type" in names
    assert "chk_hazard_reports_status" in names
