# Materiały — dowód działania (przebieg E2E)

Rzeczywisty przebieg z 2026‑06‑01 na środowisku docelowym (Windows 11, Python 3.13).

## 1. Potok offline

```
build_targets.py   -> 91 031 lokali; mediana przychodu 29 200 EUR (calendar.csv: 33 226 170 wierszy)
build_features.py  -> data/processed/model_table.csv (91 031 x 15, brak braków)
train_models.py    -> CV (poniżej) + modele na 72 096 train, 18 024 held-out test
make_ground_truth  -> data/processed/ground_truth.csv (18 024 wierszy)
```

CV (z `data/processed/cv_metrics.csv`):

| model | cv | RMSE | MAE | R² |
|---|---|---|---|---|
| A_baseline | random_kfold | 48 441 | 30 056 | 0.019 |
| A_baseline | spatial_LODO | 48 967 | 30 498 | −0.002 |
| B_hgb | random_kfold | 44 585 | 26 764 | 0.169 |
| B_hgb | spatial_LODO | 45 726 | 27 422 | 0.126 |

## 2. Serwer + przykładowe wywołania (curl.exe)

```
GET /health
{"status":"ok","models":{"a":"baseline-district-mean-2026.06.01","b":"hgb-2026.06.01"},"target":"annual_revenue"}

POST /predict_revenue            (wybór modelu PRZEZROCZYSTY — brak pola model)
{"request_id":"6d7e0f18-...","listing_id":3109,"predicted_annual_revenue":36683.32,"currency":"EUR"}

POST /predict_revenue/a          (wymuszony baseline)
{...,"predicted_annual_revenue":36683.32,"model_used":"a","model_version":"baseline-district-mean-2026.06.01"}

POST /predict_revenue/b          (wymuszony HGB)
{...,"predicted_annual_revenue":26382.63,"model_used":"b","model_version":"hgb-2026.06.01"}
```

Predykcja przez endpoint główny (36683.32) = predykcja wymuszonego modelu A → potwierdza,
że routing skierował tę ofertę do A, a klient nie widzi, który model odpowiedział.

## 3. Symulacja ruchu A/B

```
python scripts/simulate_clients.py --n 1000 --paired
Replayed 1,000 listings -> http://127.0.0.1:8080/predict_revenue
  success: 1,000   failed: 0   wall time: 255.1s
-> logs/predictions.jsonl: 3000 rekordów (1000 ruch naturalny + 2000 wymuszone /a,/b)
```

Przykładowy rekord loga: `request_id, timestamp, endpoint, assigned_model, assignment_reason,
model_version, listing_id, input_features, derived_features, predicted_annual_revenue,
latency_ms (~5 ms), ground_truth_annual_revenue=null, schema_version`.

## 4. Ewaluacja A/B (z loga)

```
=== Per-model metrics (natural A/B traffic) ===
 a  n=464  RMSE 46 280  MAE 30 261  R² 0.035  mediana AE 21 490
 b  n=536  RMSE 43 610  MAE 25 828  R² 0.170  mediana AE 16 591
Mann-Whitney p=1.8e-4   Welch p=0.047
Bootstrap 95% CI RMSE(A)-RMSE(B): [-8 486, 13 569]
Paired Wilcoxon p=1.35e-19 (n=1000)
VERDICT: WYGRYWA model B (HGB) — istotnie (parowany Wilcoxon, Mann-Whitney);
         zysk dotyczy ofert typowych (RMSE zdominowane przez ogon).
```

## 5. Testy

```
python -m pytest tests/  ->  14 passed
```

## 6. Wykresy (reports/figures/)

EDA: `eda_target_distribution.png`, `eda_revenue_by_district.png` ·
Modele: `model_cv_comparison.png`, `model_b_importance.png`, `whitespot_map.png` ·
A/B: `ab_pred_vs_actual.png`, `ab_abs_error_box.png`, `ab_metric_bars.png`.

Wykonane notatniki `notebooks/01–03` zawierają pełne wyjścia i te same wykresy.
```
