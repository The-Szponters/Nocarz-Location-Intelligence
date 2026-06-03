# Raport z eksperymentu A/B — Nocarz

Źródło: `predictions.jsonl` (ruch produkcyjny, routing hash 50/50).

## Przychód — metryki per model (ruch naturalny A/B)

| model | n | RMSE | MAE | R² | mediana AE |
|---|---|---|---|---|---|
| A (baseline / district-mean) | 928 | 46,280 | 30,261 | 0.035 | 21,490 |
| B (HGB) | 1,072 | 43,168 | 25,229 | 0.187 | 16,860 |


## Przychód — istotność statystyczna (błąd bezwzględny, testy niezależne)

- Mann-Whitney U: p = 7.136e-09
- Welch t-test: p = 0.001383
- Bootstrap 95% CI dla RMSE(A) − RMSE(B): [-4,632, 11,193] EUR

## Przychód — test parowany (wymuszone /a i /b na tych samych ofertach)

- liczba par: 4,000
- średni |błąd| A = 29,797, B = 25,313 EUR
- Wilcoxon: p = 7.003e-84; t-parowany: p = 1.289e-61

## Werdykt (przychód — główne KPI)

WYGRYWA model B (HGB): MAE 25,229 < 30,261 EUR, RMSE 43,168 < 46,280; różnica istotna (parowany Wilcoxon p=7e-84, Mann-Whitney p=7.1e-09). Rekomendacja: wdrożyć B (uwaga: luka RMSE nieistotna, 95% CI = [-4,632, 11,193] — RMSE zdominowane przez ciężki ogon błędów; przewaga B dotyczy ofert typowych).


## Obłożenie — metryki per model (drugorzędny wynik Canvas)

| model | n | RMSE | MAE | R² | mediana AE |
|---|---|---|---|---|---|
| A (baseline / district-mean) | 928 | 0.3571 | 0.3213 | 0.030 | 0.3225 |
| B (HGB) | 1,072 | 0.3473 | 0.3034 | 0.076 | 0.3024 |


- Mann-Whitney U (|błąd| obłożenia): p = 0.0004196

- Test parowany (Wilcoxon) obłożenia: p = 3.08e-33 (śr. |błąd| A=0.3223, B=0.3023)

## Wykresy

![pred vs actual](figures/ab_pred_vs_actual.png)
![abs error](figures/ab_abs_error_box.png)
![metrics](figures/ab_metric_bars.png)
![occupancy pred vs actual](figures/ab_occupancy_pred_vs_actual.png)