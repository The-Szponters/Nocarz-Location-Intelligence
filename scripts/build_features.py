"""Build the model table: listing features + spatial features + target.

Reads listings.csv, engineers the pre-launch feature set (see nocarz.features),
computes spatial competition density via the FeatureBuilder, joins the target
from listing_targets.csv, and writes data/processed/model_table.csv.

Run build_targets.py first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nocarz.features import (  # noqa: E402
    ALL_FEATURES,
    LISTINGS_CSV,
    PROCESSED_DIR,
    TARGET,
    TARGET_OCCUPANCY,
    FeatureBuilder,
    amenities_count,
    clean_price_series,
    compute_distance_features,
    count_premium_amenities,
    parse_bathrooms,
)

TARGETS_PATH = PROCESSED_DIR / "listing_targets.csv"
OUT_PATH = PROCESSED_DIR / "model_table.csv"

# Columns needed from listings.csv.
USECOLS = [
    "id",
    "latitude",
    "longitude",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "accommodates",
    "amenities",
    "bathrooms_text",
    "price",
    "review_scores_location",
]


def main() -> None:
    print(f"Loading {LISTINGS_CSV} ...")
    listings = pd.read_csv(LISTINGS_CSV, usecols=USECOLS, encoding="utf-8")
    print(f"  {len(listings):,} listings loaded")

    listings["price"] = clean_price_series(listings["price"])
    listings["amenities_count"] = listings["amenities"].map(amenities_count)
    listings["premium_amenities_count"] = listings["amenities"].map(count_premium_amenities)
    listings["bathrooms"] = listings["bathrooms_text"].map(parse_bathrooms)
    # Impute missing bath counts with the global median (modal listing is 1 bath).
    listings["bathrooms"] = listings["bathrooms"].fillna(listings["bathrooms"].median())

    # Spatial index over ALL listings; competition for each existing listing
    # excludes the listing itself.
    print("Building spatial index and computing competition density ...")
    fb = FeatureBuilder(
        listings[
            ["id", "latitude", "longitude", "neighbourhood_cleansed", "price",
             "review_scores_location"]
        ]
    )
    valid = listings.dropna(subset=["latitude", "longitude"]).copy()
    comp = fb.compute_spatial_bulk(
        valid["latitude"].to_numpy(), valid["longitude"].to_numpy()
    )
    comp.index = valid.index
    valid = pd.concat([valid, comp], axis=1)

    # Distance to key points (pure function of lat/lon; parity with serving).
    dist = compute_distance_features(
        valid["latitude"].to_numpy(), valid["longitude"].to_numpy()
    )
    for name, values in dist.items():
        valid[name] = values

    # District market aggregates.
    valid["district_median_price"] = valid["neighbourhood_cleansed"].map(
        fb.district_median_price
    )
    valid["district_price_volatility"] = (
        valid["neighbourhood_cleansed"].map(fb.district_price_volatility)
        .fillna(fb.global_price_volatility)
    )
    valid["district_mean_review_location"] = valid["neighbourhood_cleansed"].map(
        fb.district_mean_review_location
    )

    # Join the target.
    print(f"Joining target from {TARGETS_PATH} ...")
    targets = pd.read_csv(TARGETS_PATH, usecols=["id", TARGET, TARGET_OCCUPANCY])
    df = valid.merge(targets, on="id", how="inner")
    print(f"  {len(df):,} listings have a matching target")

    # Keep only listings with a usable target (had at least one calendar entry).
    before = len(df)
    df = df[df[TARGET].notna()].copy()
    print(f"  dropped {before - len(df):,} rows with missing target")

    out_cols = ["id"] + ALL_FEATURES + [TARGET, TARGET_OCCUPANCY]
    df[out_cols].to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows x {len(out_cols)} cols -> {OUT_PATH}")
    print("\nFeature null counts:")
    print(df[ALL_FEATURES].isna().sum().to_string())
    print(f"\n{TARGET} summary:")
    print(df[TARGET].describe().to_string())


if __name__ == "__main__":
    main()
