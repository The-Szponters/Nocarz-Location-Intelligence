# Raport z eksperymentu A/B — Nocarz

Źródło: `predictions.jsonl` (ruch produkcyjny, routing hash 50/50).

## Metryki per model (ruch naturalny A/B)

| model | n | RMSE | MAE | R² | mediana AE |
|---|---|---|---|---|---|
| A (baseline / district-mean) | 464 | 46,280 | 30,261 | 0.035 | 21,490 |
| B (HGB) | 536 | 43,610 | 25,828 | 0.170 | 16,591 |


## Istotność statystyczna (błąd bezwzględny, testy niezależne)

- Mann-Whitney U: p = 0.0001844
- Welch t-test: p = 0.04674
- Bootstrap 95% CI dla RMSE(A) − RMSE(B): [-8,486, 13,569] EUR

## Test parowany (wymuszone /a i /b na tych samych ofertach)

- liczba par: 1,000
- średni |błąd| A = 29,797, B = 25,712 EUR
- Wilcoxon: p = 1.354e-19; t-parowany: p = 5.128e-15

## Werdykt

WYGRYWA model B (HGB): MAE 25,828 < 30,261 EUR, RMSE 43,610 < 46,280; różnica istotna (parowany Wilcoxon p=1.4e-19, Mann-Whitney p=0.00018). Rekomendacja: wdrożyć B (uwaga: luka RMSE nieistotna, 95% CI = [-8,486, 13,569] — RMSE zdominowane przez ciężki ogon błędów; przewaga B dotyczy ofert typowych).


## Wykresy

![pred vs actual](figures/ab_pred_vs_actual.png)
![abs error](figures/ab_abs_error_box.png)
![metrics](figures/ab_metric_bars.png)