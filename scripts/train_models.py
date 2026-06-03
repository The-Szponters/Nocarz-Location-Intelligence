"""Train, validate and persist the baseline and advanced models.

Two Canvas outputs are modeled in parallel, sharing the same pre-launch feature
set (ALL_FEATURES):
* revenue   — annual_revenue (EUR), the primary business KPI.
* occupancy — booked-night fraction in [0, 1], the second Canvas output.

For each output we train:
* Model A (baseline): DistrictMeanRegressor (district-mean lookup).
* Model B (advanced): OneHot + HistGradientBoostingRegressor pipeline.

Validation follows the ML Canvas: spatial Leave-One-District-Out CV
(LeaveOneGroupOut grouped by neighbourhood_cleansed) is the honest metric;
a plain random K-fold is also reported to show how much spatial
autocorrelation inflates the optimistic (leaky) estimate. We additionally
report the Canvas success criterion in its literal, *per geographic cluster*
form (RMSE < 12% of the district's own revenue std).

Outputs: models/model_{a,b}_{revenue,occupancy}.joblib, models/registry.json,
data/processed/cv_metrics.csv, data/processed/per_cluster_metrics.csv, and
reports/figures/per_cluster_rmse_vs_std.png.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nocarz.baseline import DistrictMeanRegressor  # noqa: E402
from nocarz.features import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    REPORTS_DIR,
    TARGET,
    TARGET_OCCUPANCY,
)

MODEL_TABLE = PROCESSED_DIR / "model_table.csv"
REVENUE_CAP_Q = 0.99  # drop top-1% revenue outliers (data-quality: junk prices)
RANDOM_STATE = 42
DATE_TAG = dt.date.today().strftime("%Y.%m.%d")
FIG_DIR = REPORTS_DIR / "figures"

# The two modeled outputs and the valid prediction range used to clamp/score.
TARGETS = {"revenue": TARGET, "occupancy": TARGET_OCCUPANCY}
CLIP = {"revenue": (0.0, None), "occupancy": (0.0, 1.0)}
SUCCESS_FRACTION = 0.12  # Canvas: RMSE < 12% of std


def make_baseline() -> DistrictMeanRegressor:
    return DistrictMeanRegressor(group_col="neighbourhood_cleansed")


def make_advanced() -> Pipeline:
    """Model B: OneHot + HistGradientBoostingRegressor on the raw target.

    Note on the revenue target scale: we tried a log1p/expm1 transform (natural
    for the heavy right skew). It *lowered* MAE slightly but **hurt RMSE and R²**
    (LODO R² went negative): a log-model underpredicts the high-revenue
    listings, and raw-scale squared error — which is exactly the Canvas success
    metric — is dominated by those. We therefore keep the raw target. See
    reports/raport.md §5.
    """
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


def _metrics(y_true, y_pred, clip=(0.0, None)) -> dict:
    lo, hi = clip
    y_pred = np.clip(y_pred, lo, hi)
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


def per_cluster_report(y_true, oof, groups, clip=(0.0, None)) -> pd.DataFrame:
    """Canvas success criterion in its literal per-district form: each
    district's own RMSE vs its own target std."""
    lo, hi = clip
    d = pd.DataFrame({
        "district": np.asarray(groups),
        "y": np.asarray(y_true, dtype=float),
        "pred": np.clip(oof, lo, hi),
    })
    rows = []
    for district, sub in d.groupby("district"):
        if len(sub) < 2:
            continue
        rmse = float(mean_squared_error(sub["y"], sub["pred"]) ** 0.5)
        std = float(sub["y"].std())
        rows.append({
            "district": district,
            "n": int(len(sub)),
            "rmse": rmse,
            "std": std,
            "rmse_pct_of_std": rmse / std if std > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("rmse_pct_of_std").reset_index(drop=True)


def plot_per_cluster(pc: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(pc))))
    y = np.arange(len(pc))
    ax.barh(y, pc["rmse_pct_of_std"], color="#4C78A8")
    ax.axvline(SUCCESS_FRACTION, color="red", ls="--", lw=1.2,
               label=f"cel Canvas = {SUCCESS_FRACTION:.0%}")
    ax.set_yticks(y)
    ax.set_yticklabels(pc["district"], fontsize=7)
    ax.set_xlabel("RMSE / odch. std. dzielnicy")
    ax.set_title("Kryterium sukcesu per dzielnica (model B, LODO OOF)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    print(f"Loading {MODEL_TABLE} ...")
    df = pd.read_csv(MODEL_TABLE)
    cap = df[TARGET].quantile(REVENUE_CAP_Q)
    n_before = len(df)
    df = df[df[TARGET] <= cap].reset_index(drop=True)
    print(f"  {n_before:,} rows -> {len(df):,} after dropping revenue > {cap:,.0f} "
          f"(top {(1 - REVENUE_CAP_Q) * 100:.0f}%)")

    X = df[ALL_FEATURES]
    groups = df["neighbourhood_cleansed"]
    rev_std = float(df[TARGET].std())
    print(f"  revenue std = {rev_std:,.0f} EUR  (canvas goal: RMSE < 12% of std = "
          f"{SUCCESS_FRACTION * rev_std:,.0f})")

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logo = LeaveOneGroupOut()
    models = {"A_baseline": make_baseline, "B_hgb": make_advanced}

    rows = []
    oof_store: dict = {}
    for tgt_name, tgt_col in TARGETS.items():
        y = df[tgt_col]
        y_std = float(y.std())
        for name, factory in models.items():
            for scheme, splitter, grp in [
                ("random_kfold", kfold, None),
                ("spatial_LODO", logo, groups),
            ]:
                print(f"\n[{tgt_name}|{name}] {scheme} ...")
                oof = cross_val_oof(factory, X, y, splitter, grp)
                m = _metrics(y, oof, CLIP[tgt_name])
                m.update(target=tgt_name, model=name, cv=scheme,
                         rmse_pct_of_std=m["rmse"] / y_std)
                rows.append(m)
                oof_store[(tgt_name, name, scheme)] = oof
                print(f"  RMSE={m['rmse']:,.4f}  MAE={m['mae']:,.4f}  R2={m['r2']:.3f}  "
                      f"(RMSE/std={m['rmse'] / y_std:.2%})")

    metrics_df = pd.DataFrame(rows)[
        ["target", "model", "cv", "rmse", "mae", "r2", "rmse_pct_of_std"]
    ]
    metrics_df.to_csv(PROCESSED_DIR / "cv_metrics.csv", index=False)
    print("\n=== CV metrics ===")
    print(metrics_df.to_string(index=False))

    # --- Canvas success criterion, per geographic cluster (revenue, Model B) ---
    pc = per_cluster_report(
        df[TARGET], oof_store[("revenue", "B_hgb", "spatial_LODO")], groups,
        clip=CLIP["revenue"],
    )
    pc.to_csv(PROCESSED_DIR / "per_cluster_metrics.csv", index=False)
    frac_ok = float((pc["rmse_pct_of_std"] < SUCCESS_FRACTION).mean())
    median_ratio = float(pc["rmse_pct_of_std"].median())
    print("\n=== Per-cluster success criterion (revenue, model B, LODO) ===")
    print(f"  districts: {len(pc)}  |  median RMSE/std = {median_ratio:.1%}  |  "
          f"meeting < {SUCCESS_FRACTION:.0%}: {frac_ok:.0%}")
    plot_per_cluster(pc, FIG_DIR / "per_cluster_rmse_vs_std.png")

    # Hold out a test set for the A/B experiment so the deployed models are
    # evaluated on genuinely unseen listings. CV above (full data) is the
    # model-selection metric; deployed models are fit on the train split only.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df["neighbourhood_cleansed"],
    )
    print(f"\nTrain/test split: {len(train_df):,} train / {len(test_df):,} held-out test")
    X_train = train_df[ALL_FEATURES]

    print("Fitting deployed models (revenue + occupancy, A + B) on train split ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    deployed = {}
    for tgt_name, tgt_col in TARGETS.items():
        y_train = train_df[tgt_col]
        deployed[("a", tgt_name)] = make_baseline().fit(X_train, y_train)
        deployed[("b", tgt_name)] = make_advanced().fit(X_train, y_train)

    def _path(role: str, tgt_name: str) -> Path:
        return MODELS_DIR / f"model_{role}_{tgt_name}.joblib"

    for (role, tgt_name), est in deployed.items():
        joblib.dump(est, _path(role, tgt_name))

    # Held-out test set = A/B replay payloads + ground truth (both targets).
    req_cols = ["id", "latitude", "longitude", "neighbourhood_cleansed",
                "property_type", "room_type", "accommodates", "amenities_count",
                "bathrooms", "premium_amenities_count"]
    test_out = test_df[req_cols + [TARGET, TARGET_OCCUPANCY]].rename(
        columns={"id": "listing_id", TARGET: "true_annual_revenue",
                 TARGET_OCCUPANCY: "true_occupancy"}
    )
    test_out.to_csv(PROCESSED_DIR / "test_set.csv", index=False)
    print(f"Wrote held-out test set ({len(test_out):,} rows) -> "
          f"{PROCESSED_DIR / 'test_set.csv'}")

    def _entry(role: str, tgt_name: str, mtype: str) -> dict:
        return {
            "path": f"models/{_path(role, tgt_name).name}",
            "version": f"{mtype}-{tgt_name}-{DATE_TAG}",
            "type": mtype,
        }

    registry = {
        "a": {
            "revenue": _entry("a", "revenue", "DistrictMeanRegressor"),
            "occupancy": _entry("a", "occupancy", "DistrictMeanRegressor"),
        },
        "b": {
            "revenue": _entry("b", "revenue", "HistGradientBoostingRegressor"),
            "occupancy": _entry("b", "occupancy", "HistGradientBoostingRegressor"),
        },
        "targets": {"revenue": TARGET, "occupancy": TARGET_OCCUPANCY},
        "revenue_cap": float(cap),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2))
    print(f"Saved 4 models + registry.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
