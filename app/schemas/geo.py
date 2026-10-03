"""Conversions between PostGIS geometry values and API-friendly JSON.

Read side: ORM attributes come back as `WKBElement`; the `BeforeValidator`
types below turn them into `{latitude, longitude}` or a GeoJSON dict, so
response schemas can use `model_validate(orm_obj)` directly.

Write side: `point_to_wkb()` builds the value to assign to a geometry column.
"""

from __future__ import annotations

from typing import Annotated, Any

from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from pydantic import BaseModel, BeforeValidator, Field
from shapely.geometry import Point, mapping

SRID = 4326


class GeoPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


def point_to_wkb(latitude: float, longitude: float) -> WKBElement:
    # Note the axis order: WKT/PostGIS points are (x=longitude, y=latitude).
    return from_shape(Point(longitude, latitude), srid=SRID)


def _wkb_to_point(value: Any) -> Any:
    if isinstance(value, WKBElement):
        pt = to_shape(value)
        return {"latitude": pt.y, "longitude": pt.x}
    return value


def _wkb_to_geojson(value: Any) -> Any:
    if isinstance(value, WKBElement):
        return mapping(to_shape(value))
    return value


PointFromDb = Annotated[GeoPoint, BeforeValidator(_wkb_to_point)]
GeoJSONFromDb = Annotated[dict[str, Any], BeforeValidator(_wkb_to_geojson)]
