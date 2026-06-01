"""Train, validate and persist the baseline and advanced models.

* Model A (baseline): DistrictMeanRegressor (district-mean lookup).
* Model B (advanced): OneHot + HistGradientBoostingRegressor pipeline.

Validation follows the ML Canvas: spatial Leave-One-District-Out CV
(LeaveOneGroupOut grouped by neighbourhood_cleansed) is the honest metric;
a plain random K-fold is also reported to show how much spatial
autocorrelation inflates the optimistic (leaky) estimate.

Outputs: models/model_a_baseline.joblib, models/model_b_hgb.joblib,
models/registry.json, and metrics to data/processed/cv_metrics.csv.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nocarz.baseline import DistrictMeanRegressor  # noqa: E402
from nocarz.features import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    TARGET,
)

MODEL_TABLE = PROCESSED_DIR / "model_table.csv"
REVENUE_CAP_Q = 0.99  # drop top-1% revenue outliers (data-quality: junk prices)
RANDOM_STATE = 42
DATE_TAG = dt.date.today().strftime("%Y.%m.%d")


def make_baseline() -> DistrictMeanRegressor:
    return DistrictMeanRegressor(group_col="neighbourhood_cleansed")


def make_advanced() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


def _metrics(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, None)  # revenue cannot be negative
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_val_oof(make_model, X, y, splitter, groups=None) -> np.ndarray:
    """Out-of-fold predictions for any splitter."""
    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def main() -> None:
    print(f"Loading {MODEL_TABLE} ...")
    df = pd.read_csv(MODEL_TABLE)
    cap = df[TARGET].quantile(REVENUE_CAP_Q)
    n_before = len(df)
    df = df[df[TARGET] <= cap].reset_index(drop=True)
    print(f"  {n_before:,} rows -> {len(df):,} after dropping revenue > {cap:,.0f} "
          f"(top {(1 - REVENUE_CAP_Q) * 100:.0f}%)")

    X = df[ALL_FEATURES]
    y = df[TARGET]
    groups = df["neighbourhood_cleansed"]
    y_std = float(y.std())
    print(f"  target std = {y_std:,.0f} EUR  (canvas goal: RMSE < 12% of std = "
          f"{0.12 * y_std:,.0f})")

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logo = LeaveOneGroupOut()
    models = {"A_baseline": make_baseline, "B_hgb": make_advanced}

    rows = []
    for name, factory in models.items():
        for scheme, splitter, grp in [
            ("random_kfold", kfold, None),
            ("spatial_LODO", logo, groups),
        ]:
            print(f"\n[{name}] {scheme} ...")
            oof = cross_val_oof(factory, X, y, splitter, grp)
            m = _metrics(y, oof)
            m.update(model=name, cv=scheme, rmse_pct_of_std=m["rmse"] / y_std)
            rows.append(m)
            print(f"  RMSE={m['rmse']:,.0f}  MAE={m['mae']:,.0f}  R2={m['r2']:.3f}  "
                  f"(RMSE/std={m['rmse'] / y_std:.2%})")

    metrics_df = pd.DataFrame(rows)[
        ["model", "cv", "rmse", "mae", "r2", "rmse_pct_of_std"]
    ]
    metrics_df.to_csv(PROCESSED_DIR / "cv_metrics.csv", index=False)
    print("\n=== CV metrics ===")
    print(metrics_df.to_string(index=False))

    # Hold out a test set for the A/B experiment so the deployed models are
    # evaluated on genuinely unseen listings. CV above (full data) is the
    # model-selection metric; deployed models are fit on the train split only.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df["neighbourhood_cleansed"],
    )
    print(f"\nTrain/test split: {len(train_df):,} train / {len(test_df):,} held-out test")
    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]

    print("Fitting deployed models on the train split and persisting ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_a = make_baseline().fit(X_train, y_train)
    model_b = make_advanced().fit(X_train, y_train)
    joblib.dump(model_a, MODELS_DIR / "model_a_baseline.joblib")
    joblib.dump(model_b, MODELS_DIR / "model_b_hgb.joblib")

    # Held-out test set = A/B replay payloads + ground truth.
    req_cols = ["id", "latitude", "longitude", "neighbourhood_cleansed",
                "property_type", "room_type", "accommodates", "amenities_count"]
    test_out = test_df[req_cols + [TARGET]].rename(
        columns={"id": "listing_id", TARGET: "true_annual_revenue"}
    )
    test_out.to_csv(PROCESSED_DIR / "test_set.csv", index=False)
    print(f"Wrote held-out test set ({len(test_out):,} rows) -> "
          f"{PROCESSED_DIR / 'test_set.csv'}")

    registry = {
        "a": {
            "path": "models/model_a_baseline.joblib",
            "version": f"baseline-district-mean-{DATE_TAG}",
            "type": "DistrictMeanRegressor",
        },
        "b": {
            "path": "models/model_b_hgb.joblib",
            "version": f"hgb-{DATE_TAG}",
            "type": "HistGradientBoostingRegressor",
        },
        "target": TARGET,
        "revenue_cap": float(cap),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2))
    print(f"Saved models + registry.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
