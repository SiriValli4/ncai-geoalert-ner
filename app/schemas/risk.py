"""Request/response contract for POST /risk/predict.
Unchanged from legacy/landslide_risk_api.py so existing clients keep working.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RiskRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees.")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees.")
    rainfall_24h_mm: float = Field(..., ge=0, le=1500, description="Rainfall in the last 24h (mm).")
    soil_moisture_percent: float = Field(
        ..., ge=0, le=100, description="Volumetric soil moisture (%)."
    )
    slope_degree: float = Field(..., ge=0, le=90, description="Slope steepness in degrees.")

    # Optional overrides for terrain attributes the model needs but that aren't
    # in the required payload -- see app/services/terrain.py.
    land_cover_code: int | None = Field(
        None, ge=0, le=5, description="Optional override; looked up from lat/lon if omitted."
    )
    lithology_stability_index: int | None = Field(
        None, ge=1, le=5, description="Optional override (1=unstable - 5=stable)."
    )
    elevation_m: float | None = Field(
        None, ge=0, le=9000, description="Optional override (metres)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "latitude": 25.578,
                "longitude": 91.893,
                "rainfall_24h_mm": 210.5,
                "soil_moisture_percent": 78.0,
                "slope_degree": 42.0,
            }
        }
    }


class FactorContribution(BaseModel):
    feature: str
    value: float
    direction: Literal["increases_risk", "decreases_risk"]
    contribution_score: float = Field(
        ...,
        description="Relative strength of this factor's push on the prediction "
        "(SHAP value when available, else a heuristic score). "
        "Not directly comparable across explanation methods.",
    )
    explanation: str


class RiskResponse(BaseModel):
    latitude: float
    longitude: float
    landslide_susceptibility_index: float = Field(
        ..., description="0.0-100.0% modeled susceptibility score."
    )
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    class_probabilities: dict[str, float]
    top_factors: list[FactorContribution]
    estimated_inputs: list[str] = Field(
        ..., description="Terrain fields that were stubbed/looked up rather than supplied."
    )
    explanation_method: Literal["shap", "heuristic_importance"]
    model_type: str
