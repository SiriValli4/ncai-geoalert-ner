"""Landslide risk model: loading, inference, and explanations.

Ported from legacy/landslide_risk_api.py. The artifact is produced by
ml/training/train_landslide_risk_model.py. ML libraries are imported lazily
so the API can boot without the `risk` extra installed; the risk endpoints
then return 503.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from app.schemas.risk import FactorContribution, RiskRequest, RiskResponse
from app.services.terrain import resolve_terrain_attributes

logger = logging.getLogger(__name__)

Severity = Literal["LOW", "MEDIUM", "HIGH"]
Direction = Literal["increases_risk", "decreases_risk"]

# Model's ordinal class (0/1/2) -> susceptibility-index anchor and severity label.
CLASS_TO_LSI_ANCHOR = {0: 0.0, 1: 50.0, 2: 100.0}
CLASS_TO_SEVERITY: dict[int, Severity] = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# (low, high, direction) -- used only by the no-SHAP fallback explainer.
# direction: +1 raises risk, -1 lowers it, 0 categorical/neutral.
_FALLBACK_FEATURE_RANGES = {
    "slope_degree": (0.0, 75.0, +1),
    "rainfall_24h_mm": (0.0, 600.0, +1),
    "soil_moisture_percent": (0.0, 100.0, +1),
    "land_cover_code": (0.0, 5.0, 0),
    "lithology_stability_index": (1.0, 5.0, -1),
    "elevation_m": (0.0, 3000.0, +1),
}

# (description when pushing risk UP, description when pushing risk DOWN)
_FEATURE_DESCRIPTIONS = {
    "slope_degree": ("steep terrain", "gentle terrain"),
    "rainfall_24h_mm": ("heavy recent rainfall", "low recent rainfall"),
    "soil_moisture_percent": ("saturated soil", "dry soil"),
    "land_cover_code": ("land cover type", "land cover type"),
    "lithology_stability_index": ("unstable underlying geology", "stable underlying geology"),
    "elevation_m": ("higher elevation", "lower elevation"),
}


class ModelNotLoadedError(RuntimeError):
    pass


class RiskModelService:
    """Holds the loaded pipeline. One instance lives on `app.state.risk_model`."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.classifier: Any = None
        self.imputer: Any = None
        self.feature_columns: list[str] = []
        self.model_type: str = ""
        self.shap_explainer: Any = None
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Risk model artifact not found at '{path}'. Train one with "
                f"ml/training/train_landslide_risk_model.py or set RISK_MODEL_PATH."
            )
        import joblib

        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.imputer = self.pipeline.named_steps["imputer"]
        self.feature_columns = list(artifact["feature_columns"])
        self.model_type = artifact.get("model_type", type(self.classifier).__name__)

        try:
            import shap

            self.shap_explainer = shap.TreeExplainer(self.classifier)
        except ImportError:
            logger.info("shap not installed; using heuristic explanations")
        except Exception:
            logger.warning(
                "Could not build SHAP explainer; using heuristic explanations", exc_info=True
            )

    def try_load(self, path: Path) -> None:
        """Load, but log instead of raising -- a missing model must not stop the API."""
        try:
            self.load(path)
            logger.info("Risk model loaded (%s) from %s", self.model_type, path)
        except Exception as exc:
            self.load_error = str(exc)
            logger.warning("Risk model unavailable: %s", exc)

    # ------------------------------------------------------------------ #
    # Inference (CPU-bound and synchronous -- call from a threadpool)
    # ------------------------------------------------------------------ #
    def predict(self, request: RiskRequest) -> RiskResponse:
        if not self.is_loaded:
            raise ModelNotLoadedError(self.load_error or "Risk model is not loaded.")

        import numpy as np
        import pandas as pd

        terrain_attrs, estimated_inputs = resolve_terrain_attributes(request)
        feature_row = {
            "slope_degree": request.slope_degree,
            "rainfall_24h_mm": request.rainfall_24h_mm,
            "soil_moisture_percent": request.soil_moisture_percent,
            **terrain_attrs,
        }
        X_df = pd.DataFrame([feature_row])[self.feature_columns]
        X_imputed = self.imputer.transform(X_df)
        proba = self.classifier.predict_proba(X_imputed)[0]

        predicted_class = int(np.argmax(proba))
        classes_ = list(self.classifier.classes_)

        class_probabilities = {
            CLASS_TO_SEVERITY.get(int(c), str(c)): round(float(p), 4)
            for c, p in zip(classes_, proba, strict=True)
        }
        lsi = sum(
            float(p) * CLASS_TO_LSI_ANCHOR.get(int(c), 0.0)
            for c, p in zip(classes_, proba, strict=True)
        )
        lsi = round(min(max(lsi, 0.0), 100.0), 1)

        top_factors, method = self._build_top_factors(X_imputed, top_n=2)

        return RiskResponse(
            latitude=request.latitude,
            longitude=request.longitude,
            landslide_susceptibility_index=lsi,
            severity=CLASS_TO_SEVERITY.get(predicted_class, "MEDIUM"),
            class_probabilities=class_probabilities,
            top_factors=top_factors,
            estimated_inputs=estimated_inputs,
            explanation_method=method,
            model_type=self.model_type,
        )

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    def _explain_with_shap(self, X_imputed: Any) -> list[tuple[str, float, float]]:
        """[(feature, value, contribution_to_lsi), ...] for one row.

        Per-class SHAP values are projected onto the same 0/50/100 anchors
        used for the susceptibility index, so each contribution reads as
        "points added to / removed from the index" regardless of which class
        won. SHAP is additive across linear combinations of outputs.
        """
        import numpy as np

        shap_values = self.shap_explainer.shap_values(X_imputed)
        shap_arr = np.array(shap_values)
        anchors = np.array([CLASS_TO_LSI_ANCHOR.get(int(c), 0.0) for c in self.classifier.classes_])

        if shap_arr.ndim == 3:
            # (n_samples, n_features, n_classes) -- current shap API
            weighted = shap_arr[0] @ anchors
        elif isinstance(shap_values, list):
            # Legacy API: list of (n_samples, n_features), one per class
            stacked = np.stack([np.asarray(sv)[0] for sv in shap_values], axis=-1)
            weighted = stacked @ anchors
        else:
            weighted = shap_arr[0]

        return [
            (feat, float(X_imputed[0, i]), float(weighted[i]))
            for i, feat in enumerate(self.feature_columns)
        ]

    def _explain_with_heuristic(self, X_imputed: Any) -> list[tuple[str, float, float]]:
        """Fallback: global importances scaled by where each value sits in its
        typical range, signed by assumed risk direction. Coarser than SHAP."""
        import numpy as np

        importances = getattr(self.classifier, "feature_importances_", None)
        n = len(self.feature_columns)
        results = []
        for i, feat in enumerate(self.feature_columns):
            value = float(X_imputed[0, i])
            importance = float(importances[i]) if importances is not None else 1.0 / n
            low, high, direction = _FALLBACK_FEATURE_RANGES.get(feat, (0.0, 1.0, 0))
            normalized = 0.0 if high == low else np.clip((value - low) / (high - low), 0.0, 1.0)
            # Centred on 0.5 and scaled to ~index points, comparable-ish to SHAP.
            score = importance * direction * (normalized - 0.5) * 2 * 100
            results.append((feat, value, float(score)))
        return results

    def _build_top_factors(
        self, X_imputed: Any, top_n: int
    ) -> tuple[list[FactorContribution], Literal["shap", "heuristic_importance"]]:
        method: Literal["shap", "heuristic_importance"]
        if self.shap_explainer is not None:
            raw = self._explain_with_shap(X_imputed)
            method = "shap"
        else:
            raw = self._explain_with_heuristic(X_imputed)
            method = "heuristic_importance"

        ranked = sorted(raw, key=lambda item: abs(item[2]), reverse=True)[:top_n]

        factors = []
        for feature, value, score in ranked:
            direction: Direction = "increases_risk" if score >= 0 else "decreases_risk"
            desc_up, desc_down = _FEATURE_DESCRIPTIONS.get(feature, (feature, feature))
            desc = desc_up if direction == "increases_risk" else desc_down
            verb = (
                "pushes the susceptibility index higher"
                if direction == "increases_risk"
                else "pulls the susceptibility index lower"
            )
            factors.append(
                FactorContribution(
                    feature=feature,
                    value=round(value, 2),
                    direction=direction,
                    contribution_score=round(score, 2),
                    explanation=f"{feature} = {round(value, 2)} ({desc}) {verb} "
                    f"(~{abs(round(score, 1))} pts).",
                )
            )
        return factors, method
