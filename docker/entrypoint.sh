#!/usr/bin/env bash
# Nocarz container entrypoint — one image, several modes.
#
#   serve     (default) run the FastAPI microservice on 0.0.0.0:8080
#   pipeline  build the data/target/feature tables, train + persist both models
#   ab        run the full A/B experiment (server + replay + evaluation)
#   test      run the pytest suite
#
# Mode is the first argument, or the NOCARZ_MODE env var. Examples:
#   docker run --rm -p 8080:8080 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" nocarz
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" nocarz pipeline
#   docker run --rm -v "$PWD:/app" nocarz ab
#   docker run --rm nocarz test
set -euo pipefail

MODE="${1:-${NOCARZ_MODE:-serve}}"
HOST="${NOCARZ_HOST:-0.0.0.0}"
PORT="${NOCARZ_PORT:-8080}"

wait_for_health() {
  for _ in $(seq 1 60); do
    if python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:${PORT}/health',timeout=2)" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "server did not become healthy on port ${PORT}" >&2
  return 1
}

case "$MODE" in
  serve)
    exec uvicorn nocarz.app:app --host "$HOST" --port "$PORT" --workers 1
    ;;

  pipeline)
    # The calendar -> target step streams ~33M rows (1.4 GB) and is the slow
    # one; skip it if its output already exists unless NOCARZ_FORCE=1.
    if [[ -f data/processed/listing_targets.csv && "${NOCARZ_FORCE:-0}" != "1" ]]; then
      echo "==> listing_targets.csv exists; skipping build_targets (set NOCARZ_FORCE=1 to rebuild)"
    else
      echo "==> [1/4] build_targets.py" && python scripts/build_targets.py
    fi
    echo "==> [2/4] build_features.py"   && python scripts/build_features.py
    echo "==> [3/4] train_models.py"     && python scripts/train_models.py
    echo "==> [4/4] make_ground_truth.py" && python scripts/make_ground_truth.py
    echo "==> pipeline complete: models/ and data/processed/ are ready"
    ;;

  ab)
    echo "==> starting server in background"
    uvicorn nocarz.app:app --host 127.0.0.1 --port "$PORT" --workers 1 &
    SERVER_PID=$!
    trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
    wait_for_health
    echo "==> replaying held-out listings (paired)"
    python scripts/simulate_clients.py --n "${NOCARZ_AB_N:-1000}" --paired
    echo "==> evaluating A/B"
    python scripts/evaluate_ab.py
    echo "==> A/B report written to reports/ab_report.md"
    ;;

  test)
    exec python -m pytest tests/ -q
    ;;

  *)
    echo "unknown mode '$MODE' (expected: serve | pipeline | ab | test)" >&2
    exit 2
    ;;
esac
