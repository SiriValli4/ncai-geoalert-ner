"""Terrain attribute lookup for the risk model.

The model needs land_cover_code, lithology_stability_index and elevation_m,
which the /risk/predict payload doesn't require. This is the single place a
real GIS lookup (land-cover raster, geology layer, DEM) should be wired in.
Until then it returns deterministic placeholders, and reports which fields
were estimated so every prediction stays auditable.
"""

from __future__ import annotations

from app.schemas.risk import RiskRequest


def resolve_terrain_attributes(request: RiskRequest) -> tuple[dict[str, float], list[str]]:
    estimated: list[str] = []
    coord_seed = abs(hash((round(request.latitude, 2), round(request.longitude, 2)))) % 1000

    if request.land_cover_code is not None:
        land_cover_code = request.land_cover_code
    else:
        land_cover_code = coord_seed % 6  # TODO: real land-cover raster lookup
        estimated.append("land_cover_code")

    if request.lithology_stability_index is not None:
        lithology_stability_index = request.lithology_stability_index
    else:
        lithology_stability_index = 3  # TODO: real geological stability lookup
        estimated.append("lithology_stability_index")

    if request.elevation_m is not None:
        elevation_m = request.elevation_m
    else:
        elevation_m = 500.0 + (coord_seed * 2.5)  # TODO: real DEM lookup
        estimated.append("elevation_m")

    return (
        {
            "land_cover_code": float(land_cover_code),
            "lithology_stability_index": float(lithology_stability_index),
            "elevation_m": float(elevation_m),
        },
        estimated,
    )
