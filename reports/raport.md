# Raport — modele opłacalności lokalizacji dla portalu Nocarz

Zespół: The Szponters · Temat 12 · Dane: Paryż (Airbnb-like)

## 1. Kontekst i sformułowanie zadania

Zadanie biznesowe (enigmatyczne): *"Czasami zastanawiamy się, w którym miejscu
powinniśmy poszukać jakiegoś nowego lokalu do dodania, tak aby był jak najbardziej
opłacalny."*

Doprecyzowanie (zgodne z Machine Learning Canvas):

- **Typ zadania:** regresja punktowa w przestrzeni + post-processing (ranking, „białe plamy").
- **Predykcja:** szacowany **roczny przychód** lokalu (`annual_revenue`) dla profilu
  referencyjnego w danej lokalizacji.
- **Beneficjent:** dział Business Development — otrzymuje mapę potencjału i listę Top-N
  lokalizacji o wysokim potencjale i niskiej saturacji rynku.
- **Klucz:** predykcja musi opierać się wyłącznie na cechach znanych **przed** wystawieniem
  oferty (lokalizacja, typ lokalu, otoczenie rynkowe) — nie na wynikach po starcie.

## 2. Inżynieria zmiennej celu (Ground Truth)

Etykiety z Canvas (`estimated_revenue_l365d`, `estimated_occupancy_l365d`) oraz
`calendar_prev.csv` **nie istnieją** w danych. Cel wyznaczamy z `calendar.csv` (kalendarz
na okno ~365 dni, 2024‑12‑06 → 2025‑12‑09):

```
annual_revenue = Σ price  po nocach zarezerwowanych (available == 'f')
```

`calendar.csv` (33 mln wierszy, 1,4 GB) przetwarzamy strumieniowo w porcjach (`build_targets.py`).

**Założenie:** `available == False` traktujemy jako „zarezerwowane" (najlepszy dostępny sygnał
realizacji przychodu; w danych syntetycznych akceptowalne — odnotowane jawnie).

Cel jest silnie prawoskośny (mediana 29 200 EUR, ale wartości skrajne do 329 mln EUR —
nierealne ceny). W modelowaniu **odcinamy górny 1%** (dane śmieciowe); zera (lokale nigdy
nierezerwowane) zostają. Rozkład: `figures/eda_target_distribution.png`.

## 3. Cechy i unikanie wycieku danych

Zestaw cech (identyczny w treningu i w serwowaniu — `src/nocarz/features.py`):

| Grupa | Cechy |
|---|---|
| Geograficzne | `latitude`, `longitude`, `neighbourhood_cleansed` (20 dzielnic) |
| Typologia | `property_type`, `room_type`, `accommodates`, `amenities_count` |
| Otoczenie (przestrzenne) | gęstość konkurencji w 250/500/1000 m (`cKDTree`), mediana ceny dzielnicy, średnia ocena lokalizacji dzielnicy |

**Anty‑wyciek:** świadomie **pomijamy** własną cenę, własne oceny i obłożenie lokalu — to
wyniki po starcie, niedostępne dla nowej oferty. Gęstość konkurencji liczymy rzutując
współrzędne na metry (przybliżenie równoodległościowe) i odpytując `cKDTree` (bez geopandas).
Analiza: `figures/eda_revenue_by_district.png` — zmienność **wewnątrz** dzielnic
przewyższa zmienność **między** nimi, co z góry ogranicza model „średniej dzielnicy".

## 4. Modele

- **Model A — bazowy (najprostszy możliwy):** `DistrictMeanRegressor` — średni przychód
  w dzielnicy, z odwrotem do średniej globalnej dla nieznanych dzielnic. Czysty lookup
  przestrzenny.
- **Model B — docelowy:** `OneHotEncoder` + `HistGradientBoostingRegressor` (sklearn,
  Gradient Boosting zgodnie z Canvas; bez dodatkowych zależności).

**Walidacja (wymóg Canvas):** ze względu na autokorelację przestrzenną zwykły losowy podział
zawyża wyniki (wyciek). Stosujemy **przestrzenną CV: Leave‑One‑District‑Out**
(`LeaveOneGroupOut` po dzielnicach). Dla kontrastu raportujemy też podział losowy.

## 5. Wyniki walidacji krzyżowej

| Model | Schemat CV | RMSE [EUR] | MAE [EUR] | R² |
|---|---|---|---|---|
| A (bazowy) | losowy K‑fold | 48 441 | 30 056 | 0.019 |
| A (bazowy) | **przestrzenny LODO** | 48 967 | 30 498 | −0.002 |
| B (HGB) | losowy K‑fold | 44 585 | 26 764 | 0.169 |
| B (HGB) | **przestrzenny LODO** | 45 726 | 27 422 | **0.126** |

Wykres: `figures/model_cv_comparison.png`. Ważność cech: `figures/model_b_importance.png`.

**Interpretacja:**
- Model bazowy ma praktycznie zerową moc predykcyjną (R²≈0) — przychód zmienia się bardziej
  *wewnątrz* dzielnic niż *między* nimi. Pod LODO degeneruje się do średniej globalnej (R²≈0).
- Model B wydobywa realny sygnał lokalizacyjno‑typologiczny i — co kluczowe — **utrzymuje go
  pod uczciwą walidacją przestrzenną** (R²≈0.13, MAE niższe o ~9%, RMSE o ~7%).
- **Cel z Canvas** (RMSE < 12% odch. std., czyli < ~5 870 EUR) **nie jest osiągnięty** —
  RMSE ≈ 93% odch. std. To uczciwy wniosek: precyzyjna prognoza bezwzględnego przychodu
  *przed* startem (bez własnej ceny) jest wewnętrznie trudna. Model jest natomiast użyteczny
  do **różnicowania i rankingu lokalizacji**, co realizuje cel biznesowy.

## 6. Mapa potencjału i „białe plamy"

Na siatce nałożonej na Paryż liczymy dla profilu referencyjnego (cały lokal, 4 osoby) WPF
(predykcja przychodu modelu B) oraz saturację (liczba ofert w 500 m). **Biała plama** =
wysoki WPF (≥70 pct) i niska saturacja (≤30 pct). Wynik: mapa ciepła + lista Top‑N dla
Business Development — `figures/whitespot_map.png` (notatnik `02_modeling.ipynb`).

## 7. Wdrożenie produkcyjne — mikroserwis + eksperyment A/B

Mikroserwis FastAPI (`src/nocarz/app.py`, port 8080):

- `POST /predict_revenue` — serwowanie predykcji; wybór modelu **przezroczysty** dla klienta
  (odpowiedź nie ujawnia, który model odpowiedział).
- `GET /health`, `POST /predict_revenue/{a|b}` (debug/wymuszenie — testy i analiza parowana).
- **Routing A/B:** deterministyczny hash (sha256) klucza (`client_id` lub `listing_id`),
  50/50, „lepki" (ten sam klient → ten sam model), bez stanu. Rerandomizacja przez `SALT`,
  strojenie udziału przez `NOCARZ_AB_SPLIT`.
- **Logowanie:** każda predykcja → linia JSON (`logs/predictions.jsonl`) z `model_version`,
  cechami, predykcją, latencją i miejscem na prawdę (uzupełniane offline). Rejestr modeli
  (`models/registry.json`) umożliwia wdrażanie kolejnych wersji bez zmian w kodzie.

Predykcje serwowane są przez modele wytrenowane na **części treningowej** (80%), a eksperyment
A/B ocenia je na **wydzielonym zbiorze testowym** (18 024 ofert nieużytych w treningu).

## 8. Wyniki eksperymentu A/B (z loga mikroserwisu)

Symulacja: `simulate_clients.py` odtworzył 1000 ofert testowych (`scripts/evaluate_ab.py`).

| Model | n | RMSE | MAE | R² | mediana AE |
|---|---|---|---|---|---|
| A (bazowy) | 464 | 46 280 | 30 261 | 0.035 | 21 490 |
| B (HGB) | 536 | 43 610 | 25 828 | 0.170 | 16 591 |

- Mann‑Whitney U (błąd bezwzgl., niezależny): p = 1.8e‑4 — istotnie na korzyść B.
- **Test parowany** (te same oferty przez `/a` i `/b`, n=1000): Wilcoxon p ≈ 1.4e‑19 —
  silnie na korzyść B (najmocniejszy test, eliminuje wpływ doboru ofert).
- Bootstrap 95% CI luki RMSE(A)−RMSE(B): [−8 486, 13 569] — **nieistotne** (RMSE zdominowane
  przez ciężki ogon błędów przy ~500 obserwacjach/grupę).

**Werdykt:** wdrożyć **model B**. Przewaga jest istotna na MAE i w teście parowanym; brak
istotności na RMSE wynika z ciężkiego ogona — zysk B dotyczy ofert typowych. Pełny raport:
`reports/ab_report.md`.

## 9. Ograniczenia i dalsze kroki

- `available == False` łączy „zarezerwowane" i „zablokowane przez gospodarza".
- Bezwzględne R² jest niskie (celowe pominięcie własnej ceny) — model służy do **rankingu
  lokalizacji**, nie precyzyjnej wyceny.
- Dalej: cel w skali log, cechy popytu z `sessions.csv`, sentyment z `reviews.csv` (NLP),
  strojenie hiperparametrów, monitoring dryfu (zgodnie z Canvas).

## 10. Jak uruchomić

Patrz `README.md`. Skrót:
`build_targets.py` → `build_features.py` → `train_models.py` → `make_ground_truth.py` →
`run_server.ps1` → `simulate_clients.py` → `evaluate_ab.py` (lub notatniki `01–03`).
