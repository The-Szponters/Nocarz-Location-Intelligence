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
annual_revenue = Σ price       po nocach zarezerwowanych (available == 'f')
occupancy      = noce_zarezerwowane / wszystkie_noce   ∈ [0, 1]
```

Modelujemy **oba wyniki z Canvas**: roczny przychód (główne KPI) i obłożenie (`occupancy`).
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
| Geograficzne | `latitude`, `longitude`, `neighbourhood_cleansed` (20 dzielnic), `dist_center_km` (do Point Zéro), `dist_nearest_landmark_km` (do najbliższego z 7 kluczowych punktów) |
| Typologia / standard | `property_type`, `room_type`, `accommodates`, `amenities_count`, `bathrooms` (parsowane z `bathrooms_text`), `premium_amenities_count` (liczba kategorii udogodnień premium) |
| Otoczenie (przestrzenne) | gęstość konkurencji w 250/500/1000 m (`cKDTree`), mediana ceny dzielnicy, `district_price_volatility` (zmienność ceny dzielnicy), średnia ocena lokalizacji dzielnicy |

`bathrooms` i `premium_amenities_count` realizują postulaty Canvas (parsowanie `bathrooms_text`,
„obecność cech premium"). W serwisie są **opcjonalne** — przy braku imputujemy medianę miejską
(łazienki) i 0 (premium), więc minimalne żądanie nadal zwraca predykcję, zachowując zgodność
trening/serwowanie.

`dist_center_km` / `dist_nearest_landmark_km` realizują postulat Canvas „odległość od kluczowych
punktów" — są czystą funkcją współrzędnych (parytet trening/serwowanie zagwarantowany). Atrybut
historyczny Canvas „zmienność ceny" realizujemy jako **`district_price_volatility`** —
współczynnik zmienności ceny w dzielnicy (`std/mean`). Ponieważ `calendar_prev.csv` (szereg
czasowy) nie istnieje w danych, jest to proxy przekrojowe na poziomie dzielnicy (znane przed
startem oferty).

**Anty‑wyciek:** świadomie **pomijamy** własną cenę, własne oceny i obłożenie lokalu — to
wyniki po starcie, niedostępne dla nowej oferty. Gęstość konkurencji liczymy rzutując
współrzędne na metry (przybliżenie równoodległościowe) i odpytując `cKDTree` (bez geopandas).
Analiza: `figures/eda_revenue_by_district.png` — zmienność **wewnątrz** dzielnic
przewyższa zmienność **między** nimi, co z góry ogranicza model „średniej dzielnicy".

## 4. Modele

- **Model A — bazowy (najprostszy możliwy):** `DistrictMeanRegressor` — średnia wartość celu
  w dzielnicy, z odwrotem do średniej globalnej dla nieznanych dzielnic. Czysty lookup
  przestrzenny.
- **Model B — docelowy:** `OneHotEncoder` + `HistGradientBoostingRegressor` (sklearn,
  Gradient Boosting zgodnie z Canvas; bez dodatkowych zależności). Przychód trenowany w
  **skali surowej (EUR)** — patrz uwaga o transformacji logarytmicznej w §5.

**Oba wyniki Canvas, oba modele.** Trenujemy w sumie **cztery** modele: parę A/B dla przychodu
oraz parę A/B dla obłożenia. Wszystkie korzystają z tego samego zestawu cech `ALL_FEATURES`
(predykcja obłożenia jest przycinana do przedziału [0, 1]). Predykcja jest tania (< 1 s,
tryb on-demand z Canvas); pełną mapę potencjału generujemy wsadowo (notatnik `02_modeling.ipynb`).

**Walidacja (wymóg Canvas):** ze względu na autokorelację przestrzenną zwykły losowy podział
zawyża wyniki (wyciek). Stosujemy **przestrzenną CV: Leave‑One‑District‑Out**
(`LeaveOneGroupOut` po dzielnicach). Dla kontrastu raportujemy też podział losowy.

## 5. Wyniki walidacji krzyżowej

**Przychód (`annual_revenue`, EUR):**

| Model | Schemat CV | RMSE [EUR] | MAE [EUR] | R² |
|---|---|---|---|---|
| A (bazowy) | losowy K‑fold | 48 441 | 30 056 | 0.019 |
| A (bazowy) | **przestrzenny LODO** | 48 967 | 30 498 | −0.002 |
| B (HGB) | losowy K‑fold | 43 896 | 26 246 | 0.195 |
| B (HGB) | **przestrzenny LODO** | 45 243 | 27 125 | **0.144** |

Odch. std. przychodu (po odcięciu 1%) = 48 914 EUR.

**Obłożenie (`occupancy`, frakcja [0, 1]):**

| Model | Schemat CV | RMSE | MAE | R² |
|---|---|---|---|---|
| A (bazowy) | losowy K‑fold | 0.368 | 0.332 | 0.023 |
| A (bazowy) | **przestrzenny LODO** | 0.373 | 0.339 | −0.003 |
| B (HGB) | losowy K‑fold | 0.353 | 0.310 | 0.102 |
| B (HGB) | **przestrzenny LODO** | 0.358 | 0.316 | **0.074** |

Odch. std. obłożenia = 0.372. Wykres: `figures/model_cv_comparison.png`,
ważność cech: `figures/model_b_importance.png`.

**Interpretacja:**
- Model bazowy ma praktycznie zerową moc predykcyjną (R²≈0) dla obu celów — wartości zmieniają
  się bardziej *wewnątrz* dzielnic niż *między* nimi. Pod LODO degeneruje się do średniej globalnej.
- Model B wydobywa realny sygnał lokalizacyjno‑typologiczny i — co kluczowe — **utrzymuje go
  pod uczciwą walidacją przestrzenną** (przychód: R²≈0.14, MAE niższe o ~11%, RMSE o ~8% względem
  bazowego; obłożenie: R²≈0.07). Dodanie cech `bathrooms`, `premium_amenities_count`,
  `dist_*_km` i `district_price_volatility` oraz strojenie HGB podniosło LODO R² przychodu do 0.144.
- **Odrzucona transformacja log (przychód):** przetestowaliśmy cel w skali `log1p`/`expm1`
  (naturalny przy silnej prawoskośności). Obniżył on MAE, ale **pogorszył RMSE i R²** (LODO R²
  spadło poniżej zera): model logarytmiczny zaniża wysokie przychody, a RMSE w skali surowej —
  będące **metryką sukcesu z Canvas** — jest zdominowane właśnie przez ten ciężki ogon. Dlatego
  model docelowy trenujemy na surowym przychodzie (EUR).

**Kryterium sukcesu Canvas — w ujęciu per klaster geograficzny.** Canvas definiuje cel jako
*„RMSE < 12% odch. std. w danym klastrze geograficznym"*. Liczymy go więc dosłownie: dla każdej
z 20 dzielnic RMSE modelu B (predykcje LODO out‑of‑fold) wobec **własnego** odch. std. dzielnicy
(`per_cluster_metrics.csv`, `figures/per_cluster_rmse_vs_std.png`).

- Wynik: **mediana RMSE/std ≈ 93,5%**, a **0 z 20 dzielnic** spełnia próg < 12%. Globalny
  wskaźnik (RMSE ≈ 93% odch. std.) potwierdza się więc także lokalnie — to nie artefakt agregacji.
- **Wniosek (uczciwy):** próg 12% zakłada niemal punktową wycenę kwotową, nieosiągalną *przed*
  startem oferty bez znajomości jej własnej ceny. Model jest natomiast użyteczny do
  **różnicowania i rankingu lokalizacji**, co realizuje rzeczywisty cel biznesowy (patrz §10).

## 6. Mapa potencjału i „białe plamy"

Na siatce nałożonej na Paryż liczymy dla profilu referencyjnego (cały lokal, 4 osoby) WPF
(predykcja przychodu modelu B), przewidywane **obłożenie** (drugi model B) oraz saturację
(liczba ofert w 500 m). **Biała plama** = wysoki WPF (≥70 pct) i niska saturacja (≤30 pct);
przewidywane obłożenie wzbogaca listę Top‑N o sygnał popytu (jak szybko lokal się zapełni).
Wynik: mapa ciepła + lista Top‑N dla Business Development — `figures/whitespot_map.png`
(notatnik `02_modeling.ipynb`).

## 7. Wdrożenie produkcyjne — mikroserwis + eksperyment A/B

Mikroserwis FastAPI (`src/nocarz/app.py`, port 8080):

- `POST /predict_revenue` — serwowanie predykcji; wybór modelu **przezroczysty** dla klienta
  (odpowiedź nie ujawnia, który model odpowiedział). Odpowiedź zwraca **oba** wyniki Canvas:
  `predicted_annual_revenue` i `predicted_occupancy`.
- `GET /health`, `POST /predict_revenue/{a|b}` (debug/wymuszenie — testy i analiza parowana).
- **Routing A/B:** deterministyczny hash (sha256) klucza (`client_id` lub `listing_id`),
  50/50, „lepki" (ten sam klient → ten sam model), bez stanu. Rerandomizacja przez `SALT`,
  strojenie udziału przez `NOCARZ_AB_SPLIT`.
- **Logowanie:** każda predykcja → linia JSON (`logs/predictions.jsonl`, `schema_version=2`) z
  wersjami modeli (przychód + obłożenie), cechami, obiema predykcjami, latencją i miejscem na
  prawdę (uzupełniane offline). Rejestr modeli (`models/registry.json`, struktura
  rola→cel→wersja) umożliwia wdrażanie kolejnych wersji bez zmian w kodzie.

Predykcje serwowane są przez modele wytrenowane na **części treningowej** (80%), a eksperyment
A/B ocenia je na **wydzielonym zbiorze testowym** (18 024 ofert nieużytych w treningu).

## 8. Wyniki eksperymentu A/B (z loga mikroserwisu)

Symulacja: `simulate_clients.py` odtworzył 1000 ofert testowych (`scripts/evaluate_ab.py`).

**Przychód (główne KPI):**

| Model | n | RMSE | MAE | R² | mediana AE |
|---|---|---|---|---|---|
| A (bazowy) | 464 | 46 280 | 30 261 | 0.035 | 21 490 |
| B (HGB) | 536 | 43 168 | 25 229 | 0.187 | 16 860 |

- Mann‑Whitney U (błąd bezwzgl., niezależny): p = 4.3e‑5 — istotnie na korzyść B.
- **Test parowany** (te same oferty przez `/a` i `/b`, n=1000): Wilcoxon p ≈ 2.9e‑22 —
  silnie na korzyść B (najmocniejszy test, eliminuje wpływ doboru ofert).
- Bootstrap 95% CI luki RMSE(A)−RMSE(B): [−8 337, 14 454] — **nieistotne** (RMSE zdominowane
  przez ciężki ogon błędów przy ~500 obserwacjach/grupę).

**Obłożenie (drugi wynik Canvas):**

| Model | n | RMSE | MAE | R² | mediana AE |
|---|---|---|---|---|---|
| A (bazowy) | 464 | 0.357 | 0.321 | 0.030 | 0.323 |
| B (HGB) | 536 | 0.347 | 0.303 | 0.076 | 0.302 |

Model B jest lepszy także dla obłożenia (niższe RMSE/MAE, wyższe R²). Wykresy obu celów +
pełne testy: `reports/ab_report.md`.

**Werdykt:** wdrożyć **model B**. Przewaga jest istotna na MAE i w teście parowanym; brak
istotności na RMSE wynika z ciężkiego ogona — zysk B dotyczy ofert typowych. Pełny raport:
`reports/ab_report.md`.

## 9. Ograniczenia i dalsze kroki

- `available == False` łączy „zarezerwowane" i „zablokowane przez gospodarza".
- Bezwzględne R² jest niskie (celowe pominięcie własnej ceny) — model służy do **rankingu
  lokalizacji**, nie precyzyjnej wyceny. Dotyczy to obu celów (przychód i obłożenie).
- **Obłożenie (`occupancy`):** zrealizowane — drugi wynik Canvas jest modelowany (para A/B),
  serwowany (`predicted_occupancy`), oceniany w A/B i wykorzystany w mapie „białych plam".
  Jego moc predykcyjna jest skromna (LODO R²≈0.07), ale dodatnia i zgodna z kierunkiem przychodu.
- **`district_price_volatility`** to proxy przekrojowe — pełna „historyczna zmienność ceny" z
  Canvas wymagałaby szeregu czasowego (`calendar_prev.csv`), którego w danych brak.
- Dalej: cechy popytu z `sessions.csv`, sentyment z `reviews.csv` (NLP), pełne strojenie
  hiperparametrów, monitoring dryfu danych (zgodnie z Canvas). Transformacja log celu została
  **przetestowana i odrzucona** (§5).

## 10. Co to znaczy dla Business Development (interpretacja biznesowa)

Inwestora nie interesuje R² — interesuje go, **czy model pomaga wybierać lepsze lokalizacje niż
dotychczasowa intuicja (dzielnica)**. Tak prezentujemy skuteczność:

- **Język pieniądza zamiast statystyki.** Model B myli się średnio o **~25 200 EUR** (MAE) na
  ofertę wobec **~30 300 EUR** dla podejścia „średnia dzielnicy" — **~17% mniej błędu**. Na
  typowej (medianowej) ofercie błąd to **~16 900 EUR** wobec ~21 500 EUR (**~21% mniej**). Każda
  trafniejsza decyzja to oszczędność kosztu pozyskania oferenta w nierentownej lokalizacji.
- **To jest decyzja rankingowa, nie wycena co do euro.** Cel z Canvas (RMSE < 12% odch. std.)
  zakłada precyzyjną wycenę kwotową — to jest nieosiągalne *przed* startem oferty bez znajomości
  jej własnej ceny (potwierdza to analiza per‑dzielnica w §5). Ale zadanie biznesowe brzmi
  „**gdzie** szukać", a do uszeregowania lokalizacji model B ma realną i istotną przewagę
  (parowany Wilcoxon p ≈ 2.9e‑22).
- **Dwa sygnały, jedna decyzja.** Obok przychodu dostarczamy przewidywane **obłożenie** — dział
  widzi nie tylko *ile* lokalizacja może zarobić, ale *jak szybko* się zapełni (płynność leadu).
- **Namacalny produkt.** Wynik dla działu to nie liczba, lecz **mapa potencjału + lista Top‑N
  „białych plam"** (wysoki przewidywany przychód, niska saturacja konkurencji — §6). To gotowy
  lead do działania, a nie metryka do interpretacji.
- **Jak zmierzyć sukces po wdrożeniu (zgodnie z Canvas):** eksperyment A/B na żywym ruchu mierzy
  **Revenue Lift** lokali pozyskanych z rekomendacji vs organicznych oraz współczynnik konwersji
  w strefach „białych plam" — wprost przekładalne na przychód serwisu z prowizji.

## 11. Jak uruchomić

Patrz `README.md` — wspierana ścieżka to **Docker** (jeden obraz, tryby `pipeline | serve | ab |
test`). Skrót potoku: `pipeline` buduje cel → cechy → trenuje modele → ground truth; `serve`
uruchamia mikroserwis; `ab` wykonuje pełny eksperyment A/B i zapisuje `reports/ab_report.md`.
