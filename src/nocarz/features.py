"""Feature engineering shared by training and serving.

The same :class:`FeatureBuilder` is used to build the offline training table
(``scripts/build_features.py``) and to compute features at request time
(``app.py``). Building it from the same ``listings.csv`` on both sides
guarantees train/serve parity (no training-serving skew).

Design notes
------------
* Only *pre-launch-knowable* features are used: location, planned property
  typology, and market/neighbourhood aggregates. The listing's own price,
  reviews and occupancy are deliberately excluded (they are outcomes, not
  inputs available before a new listing is created).
* Spatial competition is computed with a planar projection of Paris
  coordinates into metres (equirectangular approximation, accurate over the
  ~10 km city span) + a ``scipy.spatial.cKDTree`` radius query. This avoids a
  geopandas/shapely dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

LISTINGS_CSV = DATA_DIR / "listings.csv"
CALENDAR_CSV = DATA_DIR / "calendar.csv"

# --- Model contract --------------------------------------------------------
# Features the model consumes. These are exactly reproducible at serve time
# from the request payload + the FeatureBuilder spatial index.
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "accommodates",
    "amenities_count",
    "comp_count_250m",
    "comp_count_500m",
    "comp_count_1000m",
    "district_median_price",
    "district_mean_review_location",
]
CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "annual_revenue"

# Competition-density radii in metres.
RADII_M = {"comp_count_250m": 250.0, "comp_count_500m": 500.0, "comp_count_1000m": 1000.0}

# Equirectangular projection constants.
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON = 111_320.0


# --- Column cleaning helpers ----------------------------------------------
def clean_price_series(s: pd.Series) -> pd.Series:
    """Convert ``$1,234.00`` style strings to float; non-parsable -> NaN."""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )


def amenities_count(value) -> int:
    """Count amenities in a JSON-array string like ``["Wifi", "Kitchen"]``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        parsed = json.loads(value)
        return len(parsed) if isinstance(parsed, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0


# --- Feature builder -------------------------------------------------------
class FeatureBuilder:
    """Builds spatial / neighbourhood features from the reference listings.

    Parameters
    ----------
    ref : DataFrame with columns ``id, latitude, longitude,
        neighbourhood_cleansed, price`` (price already numeric).
    """

    def __init__(self, ref: pd.DataFrame):
        ref = ref.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        self.ref = ref
        self.lat0 = float(ref["latitude"].mean())
        self.lon0 = float(ref["longitude"].mean())
        xy = self._project(ref["latitude"].to_numpy(), ref["longitude"].to_numpy())
        self.tree = cKDTree(xy)

        # District-level market aggregates (precomputed once).
        grp = ref.groupby("neighbourhood_cleansed")
        self.district_median_price = grp["price"].median()
        self.district_listing_count = grp.size()
        if "review_scores_location" in ref.columns:
            self.district_mean_review_location = grp["review_scores_location"].mean()
        else:
            self.district_mean_review_location = pd.Series(dtype=float)
        self.global_median_price = float(ref["price"].median())
        self.global_mean_review_location = (
            float(ref["review_scores_location"].mean())
            if "review_scores_location" in ref.columns
            else np.nan
        )

    # -- projection --
    def _project(self, lat, lon) -> np.ndarray:
        x = (np.asarray(lon) - self.lon0) * _M_PER_DEG_LON * np.cos(np.radians(self.lat0))
        y = (np.asarray(lat) - self.lat0) * _M_PER_DEG_LAT
        return np.column_stack([x, y])

    # -- bulk computation for the training table --
    def compute_spatial_bulk(self, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
        """Competition counts for points that ARE in the reference index.

        Subtracts the self-match (each listing finds itself within any radius).
        """
        xy = self._project(lat, lon)
        out = {}
        for name, r in RADII_M.items():
            counts = self.tree.query_ball_point(xy, r, return_length=True)
            out[name] = counts.astype(float) - 1.0  # exclude the self-match
        return pd.DataFrame(out)

    # -- single-point computation for serving a NEW candidate --
    def compute_spatial_point(self, lat: float, lon: float) -> dict:
        """Competition counts for a new point NOT in the index (no self-match)."""
        xy = self._project(np.array([lat]), np.array([lon]))[0]
        return {
            name: float(self.tree.query_ball_point(xy, r, return_length=True))
            for name, r in RADII_M.items()
        }

    def district_aggregates(self, district: str) -> dict:
        return {
            "district_median_price": float(
                self.district_median_price.get(district, self.global_median_price)
            ),
            "district_mean_review_location": float(
                self.district_mean_review_location.get(
                    district, self.global_mean_review_location
                )
            ),
        }

    def build_serving_row(self, f: dict) -> pd.DataFrame:
        """Turn a request feature dict into a one-row model-input DataFrame.

        ``f`` must contain: latitude, longitude, neighbourhood_cleansed,
        property_type, room_type, accommodates, amenities_count.
        Returns a DataFrame with exactly ``ALL_FEATURES`` columns.
        """
        row = {
            "latitude": f["latitude"],
            "longitude": f["longitude"],
            "accommodates": f["accommodates"],
            "amenities_count": f["amenities_count"],
            "neighbourhood_cleansed": f["neighbourhood_cleansed"],
            "property_type": f["property_type"],
            "room_type": f["room_type"],
        }
        row.update(self.compute_spatial_point(f["latitude"], f["longitude"]))
        row.update(self.district_aggregates(f["neighbourhood_cleansed"]))
        return pd.DataFrame([row])[ALL_FEATURES]

    @classmethod
    def from_listings_csv(cls, path: Path | str = LISTINGS_CSV) -> "FeatureBuilder":
        """Load the minimal reference columns from listings.csv and build."""
        ref = pd.read_csv(
            path,
            usecols=[
                "id",
                "latitude",
                "longitude",
                "neighbourhood_cleansed",
                "price",
                "review_scores_location",
            ],
            encoding="utf-8",
        )
        ref["price"] = clean_price_series(ref["price"])
        return cls(ref)
