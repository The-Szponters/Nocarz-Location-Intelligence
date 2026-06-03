# Raport z eksperymentu A/B — Nocarz

Źródło: `predictions.jsonl` (ruch produkcyjny, routing hash 50/50).

## Metryki per model (ruch naturalny A/B)

| model | n | RMSE | MAE | R² | mediana AE |
|---|---|---|---|---|---|
| A (baseline / district-mean) | 928 | 46,280 | 30,261 | 0.035 | 21,490 |
| B (HGB) | 1,072 | 42,981 | 24,988 | 0.194 | 15,681 |


## Istotność statystyczna (błąd bezwzględny, testy niezależne)

- Mann-Whitney U: p = 3.925e-10
- Welch t-test: p = 0.0007976
- Bootstrap 95% CI dla RMSE(A) − RMSE(B): [-4,468, 11,345] EUR

## Test parowany (wymuszone /a i /b na tych samych ofertach)

- liczba par: 4,000
- średni |błąd| A = 29,797, B = 25,294 EUR
- Wilcoxon: p = 5.788e-86; t-parowany: p = 8.579e-64

## Werdykt

WYGRYWA model B (HGB): MAE 24,988 < 30,261 EUR, RMSE 42,981 < 46,280; różnica istotna (parowany Wilcoxon p=5.8e-86, Mann-Whitney p=3.9e-10). Rekomendacja: wdrożyć B (uwaga: luka RMSE nieistotna, 95% CI = [-4,468, 11,345] — RMSE zdominowane przez ciężki ogon błędów; przewaga B dotyczy ofert typowych).


## Wykresy

![pred vs actual](figures/ab_pred_vs_actual.png)
![abs error](figures/ab_abs_error_box.png)
![metrics](figures/ab_metric_bars.png)