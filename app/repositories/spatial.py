"""Shared PostGIS expression helpers."""

from __future__ import annotations

from typing import Any

from sqlalchemy import ColumnElement, cast, func
from sqlalchemy.orm import QueryableAttribute

from app.db.types import PlainGeography

SRID = 4326


def make_point(latitude: float, longitude: float) -> ColumnElement[Any]:
    """`ST_SetSRID(ST_MakePoint(lng, lat), 4326)` -- note x=lng, y=lat."""
    return func.ST_SetSRID(func.ST_MakePoint(longitude, latitude), SRID)


def as_geography(expr: ColumnElement[Any] | QueryableAttribute[Any]) -> ColumnElement[Any]:
    """Cast to geography for metre-based distance. Column-side casts must be
    exactly this form to hit the `idx_*_location_geog` expression indexes."""
    return cast(expr, PlainGeography())
