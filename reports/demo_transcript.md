# Materiały — dowód działania (przebieg E2E)

Rzeczywisty przebieg z 2026‑06‑06 na środowisku docelowym (Windows 11, Python 3.12).

## 1. Potok offline

```
build_targets.py   -> 91 031 lokali; mediana przychodu 29 200 EUR (calendar.csv: 33 226 170 wierszy)
build_features.py  -> data/processed/model_table.csv (91 031 x 20, brak braków)
train_models.py    -> CV (poniżej) + modele na 72 096 train, 18 024 held-out test (po odcięciu górnego 1%: 90 120 wierszy)
make_ground_truth  -> data/processed/ground_truth.csv (18 024 wiersze)
```

CV przychodu (z `data/processed/cv_metrics.csv`; obłożenie liczone analogicznie w tym samym pliku):

| model | cv | RMSE | MAE | R² |
|---|---|---|---|---|
| A_baseline | random_kfold | 48 441 | 30 056 | 0.019 |
| A_baseline | spatial_LODO | 48 967 | 30 498 | −0.002 |
| B_hgb | random_kfold | 43 884 | 26 241 | 0.195 |
| B_hgb | spatial_LODO | 45 203 | 27 021 | 0.146 |

## 2. Serwer + przykładowe wywołania (curl.exe)

```
GET /health
{"status":"ok","models":{"a":{"revenue":"DistrictMeanRegressor-revenue-2026.06.06","occupancy":"DistrictMeanRegressor-occupancy-2026.06.06"},"b":{"revenue":"HistGradientBoostingRegressor-revenue-2026.06.06","occupancy":"HistGradientBoostingRegressor-occupancy-2026.06.06"}},"targets":{"revenue":"annual_revenue","occupancy":"occupancy"}}

POST /predict_revenue            (wybór modelu PRZEZROCZYSTY — brak pola model)
{"request_id":"f7e9f8e2-...","listing_id":3109,"predicted_annual_revenue":52265.81,"predicted_occupancy":0.5504,"currency":"EUR"}

POST /predict_revenue/a          (wymuszony baseline)
{...,"predicted_annual_revenue":52265.81,"predicted_occupancy":0.5504,"model_used":"a","model_version":"DistrictMeanRegressor-revenue-2026.06.06"}

POST /predict_revenue/b          (wymuszony HGB)
{...,"predicted_annual_revenue":52475.77,"predicted_occupancy":0.5559,"model_used":"b","model_version":"HistGradientBoostingRegressor-revenue-2026.06.06"}
```

Predykcja przez endpoint główny (52265.81) = predykcja wymuszonego modelu A → potwierdza,
że routing skierował tę ofertę do A, a klient nie widzi, który model odpowiedział.

## 3. Symulacja ruchu A/B

```
python scripts/simulate_clients.py --n 800 --paired
Replayed 800 listings -> http://127.0.0.1:8080/predict_revenue
  success: 800   failed: 0   wall time: 230.3s
-> logs/predictions.jsonl: 2400 rekordów (800 ruch naturalny + 1600 wymuszone /a,/b)
```

Przykładowy rekord loga: `request_id, timestamp, endpoint, assigned_model, assignment_reason,
model_version, occupancy_model_version, listing_id, input_features, derived_features,
predicted_annual_revenue, predicted_occupancy, latency_ms (~5–9 ms),
ground_truth_annual_revenue=null, ground_truth_occupancy=null, schema_version=2`.

## 4. Ewaluacja A/B (z loga)

```
=== Przychód — per-model metrics (natural A/B traffic) ===
 a  n=378  RMSE 47 875  MAE 31 106  R² 0.039  mediana AE 22 647
 b  n=422  RMSE 42 937  MAE 25 156  R² 0.181  mediana AE 16 964
Mann-Whitney p=9.8e-5   Welch p=0.0188
Bootstrap 95% CI RMSE(A)-RMSE(B): [-8 274, 18 013]
Paired Wilcoxon p=1.8e-15 (n=800)

=== Obłożenie — per-model metrics ===
 a  n=378  RMSE 0.360  MAE 0.325  R² 0.026
 b  n=422  RMSE 0.354  MAE 0.309  R² 0.059   (parowany Wilcoxon p=2.4e-5)

VERDICT: WYGRYWA model B (HGB) — istotnie (parowany Wilcoxon, Mann-Whitney);
         zysk dotyczy ofert typowych (RMSE zdominowane przez ogon).
```

## 5. Testy

```
python -m pytest tests/  ->  24 passed
```

## 6. Wykresy (reports/figures/)

EDA: `eda_target_distribution.png`, `eda_revenue_by_district.png`\
Modele: `model_cv_comparison.png`, `model_b_importance.png`, `per_cluster_rmse_vs_std.png`, `whitespot_map.png`\
A/B: `ab_pred_vs_actual.png`, `ab_abs_error_box.png`, `ab_metric_bars.png`, `ab_occupancy_pred_vs_actual.png`.

Wykonane notatniki `notebooks/01–03` zawierają pełne wyjścia i te same wykresy.
