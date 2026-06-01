"""Build the per-listing target table from calendar.csv (chunked).

Target definition (fixed; reused identically by make_ground_truth.py):
    annual_revenue = sum of price over BOOKED nights in the calendar window,
    where a booked night is a row with available == 'f' (False).

calendar.csv has ~33M rows (1.4 GB) so it is streamed in chunks and never
loaded whole. Output: data/processed/listing_targets.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nocarz.features import CALENDAR_CSV, PROCESSED_DIR, clean_price_series  # noqa: E402

CHUNKSIZE = 2_000_000
OUT_PATH = PROCESSED_DIR / "listing_targets.csv"


def main() -> None:
    print(f"Reading {CALENDAR_CSV} in chunks of {CHUNKSIZE:,} rows...")
    partials = []
    total_rows = 0
    reader = pd.read_csv(
        CALENDAR_CSV,
        usecols=["listing_id", "available", "price"],
        dtype={"listing_id": "int64", "available": "string", "price": "string"},
        chunksize=CHUNKSIZE,
    )
    for i, chunk in enumerate(reader, start=1):
        chunk["booked"] = chunk["available"].eq("f")
        chunk["price"] = clean_price_series(chunk["price"])
        chunk["booked_revenue"] = chunk["price"].where(chunk["booked"], 0.0)
        agg = chunk.groupby("listing_id").agg(
            booked_nights=("booked", "sum"),
            total_days=("booked", "size"),
            annual_revenue=("booked_revenue", "sum"),
        )
        partials.append(agg)
        total_rows += len(chunk)
        print(f"  chunk {i:>3}: {total_rows:>12,} rows processed")

    print("Aggregating partial results...")
    targets = (
        pd.concat(partials)
        .groupby("listing_id")
        .agg(
            booked_nights=("booked_nights", "sum"),
            total_days=("total_days", "sum"),
            annual_revenue=("annual_revenue", "sum"),
        )
        .reset_index()
        .rename(columns={"listing_id": "id"})
    )
    targets["occupancy"] = targets["booked_nights"] / targets["total_days"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(targets):,} listings -> {OUT_PATH}")
    print("\nTarget summary (annual_revenue):")
    print(targets["annual_revenue"].describe().to_string())
    print("\nOccupancy summary:")
    print(targets["occupancy"].describe().to_string())


if __name__ == "__main__":
    main()
