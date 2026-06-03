"""Demo: replay held-out listings against the running microservice.

Populates logs/predictions.jsonl, proving the service serves predictions and
routes A/B traffic. Uses only the stdlib (urllib) so no extra dependency.

Examples
--------
  python scripts/simulate_clients.py --n 800
  python scripts/simulate_clients.py --n 300 --paired   # also hit /a and /b

The server must be running (scripts/run_server.ps1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nocarz.features import PROCESSED_DIR  # noqa: E402

TEST_SET = PROCESSED_DIR / "test_set.csv"
REQ_FIELDS = ["listing_id", "latitude", "longitude", "neighbourhood_cleansed",
              "property_type", "room_type", "accommodates", "amenities_count",
              "bathrooms", "premium_amenities_count"]


def post(url: str, payload: dict, timeout: float = 10.0) -> tuple[int, dict | None]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except urllib.error.URLError as e:
        print(f"  connection error: {e.reason} — is the server running?")
        return 0, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="number of listings to replay")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="use 127.0.0.1 not localhost (avoids slow IPv6 fallback on Windows)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--paired", action="store_true",
                    help="also call /predict_revenue/a and /b for each listing")
    args = ap.parse_args()

    df = pd.read_csv(TEST_SET)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    main_url = f"{args.base_url}/predict_revenue"

    ok, fail, latencies = 0, 0, []
    t0 = time.perf_counter()
    for _, row in sample.iterrows():
        payload = {"features": {k: row[k] for k in REQ_FIELDS}}
        # Cast numpy types to native python for JSON serialization.
        f = payload["features"]
        f["listing_id"] = int(f["listing_id"])
        f["accommodates"] = int(f["accommodates"])
        f["amenities_count"] = int(f["amenities_count"])
        f["premium_amenities_count"] = int(f["premium_amenities_count"])
        f["latitude"] = float(f["latitude"]); f["longitude"] = float(f["longitude"])
        f["bathrooms"] = float(f["bathrooms"])

        status, body = post(main_url, payload)
        if status == 200:
            ok += 1
            latencies.append(body.get("request_id") is not None)
        else:
            fail += 1
            if status == 0:
                return

        if args.paired:
            post(f"{main_url}/a", payload)
            post(f"{main_url}/b", payload)

    elapsed = time.perf_counter() - t0
    print(f"\nReplayed {len(sample):,} listings -> {main_url}")
    print(f"  success: {ok:,}   failed: {fail:,}   wall time: {elapsed:.1f}s")
    if args.paired:
        print("  (paired: also logged forced /a and /b for each listing)")
    print("\nInspect the log:  Get-Content logs/predictions.jsonl -Tail 3")
    print("Evaluate:         python scripts/evaluate_ab.py")


if __name__ == "__main__":
    main()
