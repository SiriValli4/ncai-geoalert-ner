"""Custom column types."""

from __future__ import annotations

from sqlalchemy.types import UserDefinedType


class PlainGeography(UserDefinedType):
    """Cast target that renders as bare `geography`.

    GeoAlchemy2's `Geography` renders as `geography(GEOMETRY,-1)` in a CAST,
    which does not match the `(location::geography)` expression indexes, so
    Postgres would silently fall back to a sequential scan. Use this only as
    a CAST target, never as a column type.
    """

    cache_ok = True

    def get_col_spec(self, **kw: object) -> str:
        return "geography"
