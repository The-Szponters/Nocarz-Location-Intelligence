"""Produce the canonical ground-truth lookup for A/B evaluation.

Extracts listing_id -> true_annual_revenue from the held-out test set written
by train_models.py. The A/B evaluator joins the microservice log against this
on listing_id. Kept as a separate artifact so evaluation never needs the
feature columns or the models.

Run train_models.py first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nocarz.features import PROCESSED_DIR  # noqa: E402

TEST_SET = PROCESSED_DIR / "test_set.csv"
OUT_PATH = PROCESSED_DIR / "ground_truth.csv"


def main() -> None:
    df = pd.read_csv(TEST_SET, usecols=["listing_id", "true_annual_revenue"])
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} ground-truth rows -> {OUT_PATH}")
    print(df["true_annual_revenue"].describe().to_string())


if __name__ == "__main__":
    main()
