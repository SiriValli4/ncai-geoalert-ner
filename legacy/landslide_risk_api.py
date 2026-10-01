#!/usr/bin/env python3
"""
landslide_risk_api.py

A FastAPI service that loads the trained Landslide Risk model artifact
(produced by train_landslide_risk_model.py) and exposes:

    POST /predict-risk
        Input:  latitude, longitude, rainfall_24h_mm, soil_moisture_percent,
                slope_degree (+ optional terrain overrides -- see below).
        Output: a 0.0-100.0% Landslide Susceptibility Index, a severity label
                (LOW / MEDIUM / HIGH), and an explainable-AI breakdown of the
                top factors driving that specific prediction.

    GET /health
        Basic liveness + "is the model loaded" check.

Run it with:
    uvicorn landslide_risk_api:app --reload --port 8000

Environment variables:
    LANDSLIDE_MODEL_PATH   Path to the .joblib artifact
                            (default: landslide_risk_model.joblib)

--------------------------------------------------------------------------
A note on the input schema
--------------------------------------------------------------------------
The trained model was fit on six terrain/weather features: slope_degree,
rainfall_24h_mm, soil_moisture_percent, land_cover_code,
lithology_stability_index, and elevation_m. The /predict-risk request body
only carries latitude, longitude, rainfall_24h_mm, soil_moisture_percent,
and slope_degree, per spec -- latitude/longitude are not model features,
and land_cover_code / lithology_stability_index / elevation_m are not in
the payload.

In a real deployment those three terrain attributes would come from a
GIS lookup keyed on latitude/longitude (a land-cover raster, a geological
stability layer, a DEM for elevation). `resolve_terrain_attributes()`
below is that lookup point: it is stubbed with deterministic placeholder
values so the API runs standalone, but the request body also accepts
optional `land_cover_code`, `lithology_stability_index`, and `elevation_m`
overrides so a caller (or a real GIS layer wired into that function) can
supply real values. The response's `estimated_inputs` field always tells
you which terrain attributes were stubbed vs. supplied, so predictions
stay auditable. Replace `resolve_terrain_attributes()` with a real raster/
database lookup before using this in production.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


MODEL_PATH = os.environ.get("LANDSLIDE_MODEL_PATH", "landslide_risk_model.joblib")

# Maps the model's ordinal class (0/1/2) to a susceptibility-index anchor
# and to the uppercase severity label the API returns.
CLASS_TO_LSI_ANCHOR = {0: 0.0, 1: 50.0, 2: 100.0}
CLASS_TO_SEVERITY: dict[int, Literal["LOW", "MEDIUM", "HIGH"]] = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}

# Coarse domain ranges used only by the no-SHAP fallback explainer, to
# normalize each feature's contribution when SHAP isn't installed.
_FALLBACK_FEATURE_RANGES = {
    "slope_degree": (0.0, 75.0, +1),               # (low, high, direction: +1 raises risk)
    "rainfall_24h_mm": (0.0, 600.0, +1),
    "soil_moisture_percent": (0.0, 100.0, +1),
    "land_cover_code": (0.0, 5.0, 0),               # categorical / non-monotonic -> neutral
    "lithology_stability_index": (1.0, 5.0, -1),     # higher = more stable = lower risk
    "elevation_m": (0.0, 3000.0, +1),
}

# (description when this feature is pushing risk UP, description when it's pushing risk DOWN)
_FEATURE_DESCRIPTIONS = {
    "slope_degree": ("steep terrain", "gentle terrain"),
    "rainfall_24h_mm": ("heavy recent rainfall", "low recent rainfall"),
    "soil_moisture_percent": ("saturated soil", "dry soil"),
    "land_cover_code": ("land cover type", "land cover type"),
    "lithology_stability_index": ("unstable underlying geology", "stable underlying geology"),
    "elevation_m": ("higher elevation", "lower elevation"),
}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
class ModelState:
    """Holds everything loaded once at startup and reused per-request."""

    pipeline = None
    classifier = None
    imputer = None
    feature_columns: list[str] = []
    class_labels: dict[int, str] = {}
    model_type: str = ""
    shap_explainer = None


model_state = ModelState()


def load_model(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. Train one first with "
            f"train_landslide_risk_model.py, or set LANDSLIDE_MODEL_PATH."
        )
    artifact = joblib.load(path)
    model_state.pipeline = artifact["pipeline"]
    model_state.classifier = model_state.pipeline.named_steps["classifier"]
    model_state.imputer = model_state.pipeline.named_steps["imputer"]
    model_state.feature_columns = artifact["feature_columns"]
    model_state.class_labels = artifact["class_labels"]
    model_state.model_type = artifact.get("model_type", type(model_state.classifier).__name__)

    if _HAS_SHAP:
        try:
            model_state.shap_explainer = shap.TreeExplainer(model_state.classifier)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Could not build SHAP explainer, falling back to "
                  f"heuristic explanations: {exc}")
            model_state.shap_explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="NER Landslide Risk Prediction API",
    description="Predicts landslide susceptibility for the North Eastern "
                 "Region of India from weather and terrain parameters.",
    version="1.0.0",
    lifespan=lifespan,
    root_path=""
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class RiskRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees.")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees.")
    rainfall_24h_mm: float = Field(..., ge=0, le=1500, description="Rainfall in the last 24h (mm).")
    soil_moisture_percent: float = Field(..., ge=0, le=100, description="Volumetric soil moisture (%).")
    slope_degree: float = Field(..., ge=0, le=90, description="Slope steepness in degrees.")

    # Optional overrides for the terrain attributes the model also needs but
    # that aren't part of the required payload -- see module docstring.
    land_cover_code: Optional[int] = Field(
        None, ge=0, le=5, description="Optional override; looked up from lat/lon if omitted."
    )
    lithology_stability_index: Optional[int] = Field(
        None, ge=1, le=5, description="Optional override (1=unstable - 5=stable); looked up if omitted."
    )
    elevation_m: Optional[float] = Field(
        None, ge=0, le=9000, description="Optional override (metres); looked up from lat/lon if omitted."
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
        ..., description="Relative strength of this factor's push on the prediction "
                          "(SHAP value when available, else a heuristic score). "
                          "Not directly comparable across explanation methods."
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
        ..., description="Terrain fields that were stubbed/looked up rather than supplied by the caller."
    )
    explanation_method: Literal["shap", "heuristic_importance"]
    model_type: str


# --------------------------------------------------------------------------- #
# Terrain lookup stub (replace with a real GIS/DEM/geology query)
# --------------------------------------------------------------------------- #
def resolve_terrain_attributes(request: RiskRequest) -> tuple[dict[str, float], list[str]]:
    """Fill in land_cover_code / lithology_stability_index / elevation_m.

    Uses the caller's overrides when supplied; otherwise falls back to a
    deterministic placeholder derived from lat/lon so the same location
    always resolves the same way in this demo. Swap the "else" branches
    for real raster/database lookups in production.
    """
    estimated: list[str] = []
    coord_seed = abs(hash((round(request.latitude, 2), round(request.longitude, 2)))) % 1000

    if request.land_cover_code is not None:
        land_cover_code = request.land_cover_code
    else:
        land_cover_code = coord_seed % 6  # TODO: replace with a real land-cover raster lookup
        estimated.append("land_cover_code")

    if request.lithology_stability_index is not None:
        lithology_stability_index = request.lithology_stability_index
    else:
        lithology_stability_index = 3  # TODO: replace with a real geological stability lookup
        estimated.append("lithology_stability_index")

    if request.elevation_m is not None:
        elevation_m = request.elevation_m
    else:
        elevation_m = 500.0 + (coord_seed * 2.5)  # TODO: replace with a real DEM lookup
        estimated.append("elevation_m")

    return (
        {
            "land_cover_code": float(land_cover_code),
            "lithology_stability_index": float(lithology_stability_index),
            "elevation_m": float(elevation_m),
        },
        estimated,
    )


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def explain_with_shap(X_imputed: np.ndarray) -> list[tuple[str, float, float]]:
    """Returns [(feature_name, feature_value, contribution_to_lsi), ...] for one row.

    Rather than explaining a single class's probability (whose sign would
    flip depending on which class happens to win -- e.g. a feature pushing
    toward LOW looks "positive" for class 0 but that's a risk-*reducing*
    push), each feature's per-class SHAP values are projected onto the same
    0/50/100 anchor weights used to compute the Landslide Susceptibility
    Index. SHAP contributions are additive across a linear combination of
    outputs, so this yields "how many points did this feature add to (or
    subtract from) the susceptibility index" -- a direction that's always
    meaningful relative to actual risk, regardless of the predicted class.
    """
    shap_values = model_state.shap_explainer.shap_values(X_imputed)
    shap_arr = np.array(shap_values)
    classes_ = list(model_state.classifier.classes_)
    anchors = np.array([CLASS_TO_LSI_ANCHOR.get(int(c), 0.0) for c in classes_])

    if shap_arr.ndim == 3:
        # (n_samples, n_features, n_classes) -- current shap unified API
        weighted = shap_arr[0] @ anchors  # -> (n_features,)
    elif isinstance(shap_values, list):
        # Legacy API: list of (n_samples, n_features) arrays, one per class
        stacked = np.stack([np.asarray(sv)[0] for sv in shap_values], axis=-1)  # (n_features, n_classes)
        weighted = stacked @ anchors
    else:
        # Binary/regression-style single array; already on the raw output scale
        weighted = shap_arr[0]

    return [
        (feat, float(X_imputed[0, i]), float(weighted[i]))
        for i, feat in enumerate(model_state.feature_columns)
    ]


def explain_with_heuristic(X_imputed: np.ndarray) -> list[tuple[str, float, float]]:
    """Fallback local explanation: global feature_importances_ scaled by how
    far each value sits in its typical range, signed by an assumed risk
    direction. Coarser than SHAP -- install `shap` for proper local
    attributions.
    """
    importances = getattr(model_state.classifier, "feature_importances_", None)
    results = []
    for i, feat in enumerate(model_state.feature_columns):
        value = float(X_imputed[0, i])
        importance = float(importances[i]) if importances is not None else 1.0 / len(model_state.feature_columns)
        low, high, direction = _FALLBACK_FEATURE_RANGES.get(feat, (0.0, 1.0, 0))
        normalized = 0.0 if high == low else np.clip((value - low) / (high - low), 0.0, 1.0)
        # Center around 0.5 so "typical" values contribute ~0, extremes contribute more
        # Scale to roughly the same order of magnitude as the SHAP path
        # (points on the 0-100 susceptibility index), so the two methods
        # are at least visually comparable even though neither is exact.
        score = importance * direction * (normalized - 0.5) * 2 * 100
        results.append((feat, value, float(score)))
    return results


def build_top_factors(X_imputed: np.ndarray, top_n: int = 2) -> tuple[list[FactorContribution], str]:
    if model_state.shap_explainer is not None:
        raw = explain_with_shap(X_imputed)
        method = "shap"
    else:
        raw = explain_with_heuristic(X_imputed)
        method = "heuristic_importance"

    ranked = sorted(raw, key=lambda item: abs(item[2]), reverse=True)[:top_n]

    factors = []
    for feature, value, score in ranked:
        direction: Literal["increases_risk", "decreases_risk"] = (
            "increases_risk" if score >= 0 else "decreases_risk"
        )
        desc_up, desc_down = _FEATURE_DESCRIPTIONS.get(feature, (feature, feature))
        desc = desc_up if direction == "increases_risk" else desc_down
        verb = "pushes the susceptibility index higher" if direction == "increases_risk" else "pulls the susceptibility index lower"
        factors.append(
            FactorContribution(
                feature=feature,
                value=round(value, 2),
                direction=direction,
                contribution_score=round(score, 2),
                explanation=f"{feature} = {round(value, 2)} ({desc}) {verb} (~{abs(round(score, 1))} pts).",
            )
        )
    return factors, method


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if model_state.pipeline is not None else "model_not_loaded",
        "model_type": model_state.model_type,
        "shap_available": model_state.shap_explainer is not None,
    }


@app.post("/predict-risk", response_model=RiskResponse)
def predict_risk(request: RiskRequest) -> RiskResponse:
    if model_state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    terrain_attrs, estimated_inputs = resolve_terrain_attributes(request)

    feature_row = {
        "slope_degree": request.slope_degree,
        "rainfall_24h_mm": request.rainfall_24h_mm,
        "soil_moisture_percent": request.soil_moisture_percent,
        **terrain_attrs,
    }
    try:
        X_df = pd.DataFrame([feature_row])[model_state.feature_columns]
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Feature mismatch with loaded model: {exc}")

    X_imputed = model_state.imputer.transform(X_df)

    try:
        proba = model_state.classifier.predict_proba(X_imputed)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    predicted_class = int(np.argmax(proba))
    classes_ = list(model_state.classifier.classes_)

    class_probabilities = {
        CLASS_TO_SEVERITY.get(int(c), str(c)): round(float(p), 4)
        for c, p in zip(classes_, proba)
    }

    lsi = sum(
        float(p) * CLASS_TO_LSI_ANCHOR.get(int(c), 0.0)
        for c, p in zip(classes_, proba)
    )
    lsi = round(min(max(lsi, 0.0), 100.0), 1)

    severity = CLASS_TO_SEVERITY.get(predicted_class, "MEDIUM")

    top_factors, method = build_top_factors(X_imputed, top_n=2)

    return RiskResponse(
        latitude=request.latitude,
        longitude=request.longitude,
        landslide_susceptibility_index=lsi,
        severity=severity,
        class_probabilities=class_probabilities,
        top_factors=top_factors,
        estimated_inputs=estimated_inputs,
        explanation_method=method,
        model_type=model_state.model_type,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("landslide_risk_api:app", host="0.0.0.0", port=8000, reload=True)
