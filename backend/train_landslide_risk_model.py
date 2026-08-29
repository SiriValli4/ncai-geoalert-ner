#!/usr/bin/env python3
"""
train_landslide_risk_model.py

Trains a Landslide Risk Prediction model for the North Eastern Region (NER) of
India using scikit-learn (RandomForestClassifier) or XGBoost (XGBClassifier,
if the `xgboost` package is installed).

Input CSV schema (column names are configurable via CLI flags, defaults shown):
    slope_degree               float   - slope steepness in degrees
    rainfall_24h_mm             float   - cumulative rainfall in the last 24h (mm)
    soil_moisture_percent       float   - volumetric soil moisture (%)
    land_cover_code              int    - categorical land-cover class code
    lithology_stability_index    int    - rock/soil stability index, 1 (unstable) - 5 (stable)
    elevation_m                 float   - elevation above sea level (m)
    risk_level                   int    - target: 0 = Low, 1 = Medium, 2 = High

Outputs:
    * A joblib artifact (default: landslide_risk_model.joblib) containing a
      dict with the fitted sklearn Pipeline, feature names, class labels, and
      basic training metadata.
    * Console output: hold-out classification report, confusion matrix, and
      feature importances (sorted, most influential first) so the model's
      logic can be explained/audited.

Usage
-----
Train on your own data:
    python train_landslide_risk_model.py --input data/ner_landslide.csv \
        --output models/landslide_risk_model.joblib --model rf

Try it instantly with a synthetic sample dataset (no real data needed):
    python train_landslide_risk_model.py --generate-sample --input sample_data.csv \
        --output landslide_risk_model.joblib --model rf

Use XGBoost instead of RandomForest (requires `pip install xgboost`):
    python train_landslide_risk_model.py --input data/ner_landslide.csv --model xgb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# XGBoost is optional. The script falls back to RandomForest with a warning
# if it is requested but not installed.
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


FEATURE_COLUMNS = [
    "slope_degree",
    "rainfall_24h_mm",
    "soil_moisture_percent",
    "land_cover_code",
    "lithology_stability_index",
    "elevation_m",
]
TARGET_COLUMN = "risk_level"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


# --------------------------------------------------------------------------- #
# Synthetic sample data generator (handy for a first end-to-end dry run)
# --------------------------------------------------------------------------- #
def generate_sample_dataset(n_rows: int, random_state: int) -> pd.DataFrame:
    """Generate a synthetic-but-plausible NER landslide dataset.

    Ranges are loosely inspired by terrain typical of NE India (Assam,
    Meghalaya, Mizoram, Manipur, Nagaland hill tracts): steep slopes, high
    monsoon rainfall intensities, and elevations from the Brahmaputra plains
    up into the hill ranges. This is for pipeline testing/demo only -- it is
    NOT a substitute for real field/remote-sensing data.
    """
    rng = np.random.default_rng(random_state)

    slope_degree = np.clip(rng.normal(28, 12, n_rows), 0, 75)
    rainfall_24h_mm = np.clip(rng.gamma(shape=2.0, scale=45, size=n_rows), 0, 600)
    soil_moisture_percent = np.clip(rng.normal(45, 15, n_rows), 5, 100)
    land_cover_code = rng.integers(0, 6, n_rows)  # e.g. 0=forest,1=cropland,2=barren,3=built-up,4=grassland,5=water-adjacent
    lithology_stability_index = rng.integers(1, 6, n_rows)  # 1 (weak) - 5 (stable)
    elevation_m = np.clip(rng.normal(900, 500, n_rows), 20, 3000)

    df = pd.DataFrame(
        {
            "slope_degree": slope_degree.round(2),
            "rainfall_24h_mm": rainfall_24h_mm.round(1),
            "soil_moisture_percent": soil_moisture_percent.round(1),
            "land_cover_code": land_cover_code.astype(int),
            "lithology_stability_index": lithology_stability_index.astype(int),
            "elevation_m": elevation_m.round(1),
        }
    )

    # Construct a physically-motivated risk score, then discretize into
    # 0/1/2 so the synthetic labels are learnable but noisy (like real life).
    risk_score = (
        0.05 * df["slope_degree"]
        + 0.015 * df["rainfall_24h_mm"]
        + 0.03 * df["soil_moisture_percent"]
        - 0.6 * df["lithology_stability_index"]
        + 0.0015 * df["elevation_m"]
        + rng.normal(0, 1.5, n_rows)  # noise
    )
    low_thr, high_thr = np.quantile(risk_score, [0.45, 0.80])
    risk_level = np.select(
        [risk_score <= low_thr, risk_score <= high_thr],
        [0, 1],
        default=2,
    )
    df[TARGET_COLUMN] = risk_level.astype(int)
    return df


# --------------------------------------------------------------------------- #
# Training pipeline
# --------------------------------------------------------------------------- #
def build_model(model_type: str, random_state: int):
    """Return an unfitted classifier for the requested model type."""
    if model_type == "xgb":
        if not _HAS_XGBOOST:
            print(
                "[WARN] --model xgb was requested but the `xgboost` package "
                "is not installed. Falling back to RandomForestClassifier.\n"
                "        Install it with: pip install xgboost",
                file=sys.stderr,
            )
            model_type = "rf"
        else:
            return XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )

    # Default / fallback: RandomForest
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}. "
            f"Expected columns: {required_cols}"
        )

    # Basic type coercion / sanity checks
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=required_cols)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[INFO] Dropped {n_dropped} row(s) with missing/non-numeric values.")

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1, 2}):
        bad = sorted(set(df[TARGET_COLUMN].unique()) - {0, 1, 2})
        raise ValueError(
            f"risk_level must only contain 0 (Low), 1 (Medium), 2 (High). "
            f"Found unexpected value(s): {bad}"
        )
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    if len(df) < 20:
        print(
            f"[WARN] Only {len(df)} usable rows after cleaning -- results "
            "will be unreliable. Provide more labeled data for a real model.",
            file=sys.stderr,
        )

    return df


def print_feature_importances(model, feature_names: list[str]) -> None:
    if not hasattr(model, "feature_importances_"):
        print("[WARN] This model type does not expose feature_importances_.")
        return

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    print("\nFeature importances (most influential first):")
    print("-" * 46)
    for rank, idx in enumerate(order, start=1):
        print(f"{rank}. {feature_names[idx]:<28s} {importances[idx]:.4f}")
    print("-" * 46)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Landslide Risk Prediction model for NE India."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("ner_landslide_data.csv"),
        help="Path to the training CSV (default: ner_landslide_data.csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("landslide_risk_model.joblib"),
        help="Path to save the trained model artifact (default: landslide_risk_model.joblib).",
    )
    parser.add_argument(
        "--model", choices=["rf", "xgb"], default="rf",
        help="Model type: 'rf' (RandomForestClassifier, default) or "
             "'xgb' (XGBClassifier, requires xgboost to be installed).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate a synthetic sample dataset at --input (overwriting it "
             "if present) before training. Useful for a first dry run when "
             "you don't yet have real labeled data.",
    )
    parser.add_argument(
        "--sample-rows", type=int, default=2000,
        help="Number of rows to generate when --generate-sample is set (default: 2000).",
    )
    args = parser.parse_args()

    if args.generate_sample:
        print(f"[INFO] Generating {args.sample_rows} synthetic sample rows at '{args.input}' ...")
        sample_df = generate_sample_dataset(args.sample_rows, args.random_state)
        sample_df.to_csv(args.input, index=False)
        print("[INFO] Sample dataset written. NOTE: this is synthetic data for "
              "pipeline testing only, not real landslide observations.\n")

    if not args.input.exists():
        parser.error(
            f"Input file '{args.input}' not found. Pass --input <path_to_csv> "
            f"or use --generate-sample to create a synthetic test dataset."
        )

    print(f"[INFO] Loading dataset from '{args.input}' ...")
    df = load_dataset(args.input)
    print(f"[INFO] Loaded {len(df)} usable rows.")
    print("[INFO] Class distribution:")
    print(df[TARGET_COLUMN].map(RISK_LABELS).value_counts().to_string())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    classifier = build_model(args.model, args.random_state)
    model_label = type(classifier).__name__

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )

    print(f"\n[INFO] Training {model_label} on {len(X_train)} rows "
          f"(holding out {len(X_test)} rows for evaluation) ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print(f"Hold-out evaluation ({model_label})")
    print("=" * 60)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}\n")
    print("Classification report:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=[RISK_LABELS[c] for c in sorted(RISK_LABELS)],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows = actual, cols = predicted; order Low, Medium, High):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    print_feature_importances(pipeline.named_steps["classifier"], FEATURE_COLUMNS)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_labels": RISK_LABELS,
        "model_type": model_label,
        "trained_on_rows": len(X_train),
        "sklearn_metrics": {"accuracy": acc, "macro_f1": macro_f1},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"\n[INFO] Saved trained model artifact to '{args.output}'.")
    print(
        "        Load it later with:\n"
        "        artifact = joblib.load('"
        f"{args.output}"
        "')\n"
        "        model = artifact['pipeline']\n"
        "        model.predict(new_data_df[artifact['feature_columns']])"
    )


if __name__ == "__main__":
    main()
