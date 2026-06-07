# Nocarz — modele opłacalności lokalizacji + mikroserwis A/B

Projekt IUM (temat 12), zespół **The Szponters**.

Wcielamy się w analityków portalu **Nocarz** (najem krótkoterminowy). Dział Business Development
pyta: *„w którym miejscu szukać nowego lokalu, aby był jak najbardziej opłacalny?"*. Budujemy
model, który na podstawie cech **znanych przed wystawieniem oferty** (lokalizacja, typ lokalu,
otoczenie rynkowe) prognozuje oba wyniki z Canvas — **roczny przychód** lokalu (`annual_revenue`,
EUR) oraz **obłożenie** (`occupancy`) — a także mikroserwis, który serwuje predykcje za
przezroczystym eksperymentem **A/B** dwóch modeli i loguje dane do jego późniejszej oceny.

Pełna metodyka i wyniki: **[`reports/raport.md`](reports/raport.md)** - raport A/B:
**[`reports/ab_report.md`](reports/ab_report.md)**.

## Architektura

Dwie połowy systemu współdzielą **jeden moduł cech** (`src/nocarz/features.py`), co gwarantuje
**zgodność trening/serwowanie** (brak train/serve skew): ten sam `FeatureBuilder`, zbudowany z tego
samego `listings.csv`, liczy cechy zarówno offline (trening), jak i przy każdym żądaniu.

```
src/nocarz/      kod współdzielony: features (kontrakt modelu), schematy, routing A/B,
                 logowanie, rejestr modeli, aplikacja FastAPI
scripts/         potok offline (build_targets → build_features → train_models →
                 make_ground_truth) + symulator klientów + ewaluacja A/B
docker/          entrypoint.sh — jeden obraz, tryby: serve | pipeline | ab | test
notebooks/       01_eda - 02_modeling (mapa „białych plam") - 03_ab_evaluation
models/          zapisane modele + registry.json (rejestr wdrożonych wersji)
data/            listings.csv, calendar.csv (wejście) + data/processed/ (artefakty)
reports/         raport.md, ab_report.md, figures/
```

- **Model A (bazowy):** `DistrictMeanRegressor` — średnia wartość celu w dzielnicy (czysty lookup).
- **Model B (docelowy):** `OneHotEncoder` + `HistGradientBoostingRegressor`.
- **Dwa cele, cztery modele:** każdy z wyników (przychód, obłożenie) ma własną parę A/B; wszystkie
  dzielą ten sam zestaw cech. `registry.json` ma strukturę rola→cel→wersja.
- **Routing A/B:** deterministyczny hash SHA‑256 klucza (`client_id` lub `listing_id`), 50/50,
  „lepki" i bezstanowy. Udział strojony przez `NOCARZ_AB_SPLIT`.
- **Kontrakt:** jako *wejścia* używamy wyłącznie cech sprzed startu oferty; własna cena/oceny są
  świadomie pominięte (to wyniki). Obłożenie jest *celem* predykcji, nie wejściem.

## Wymagania

- **Docker** (obraz oparty na `python:3.12-slim`).
- Dane wejściowe w `data/`: **`listings.csv`** i **`calendar.csv`**.
  Są duże i nie są wersjonowane — montujemy je jako wolumen, nie wbudowujemy w obraz.

Spójne środowisko 3.12 daje powtarzalne, identyczne wyniki i eliminuje problem deserializacji
modeli między wersjami Pythona (`PCG64 is not a known BitGenerator`).

## Szybki start (Docker)

Zbuduj obraz raz:

```bash
docker build -t nocarz .
```

Obraz ma **cztery tryby** (pierwszy argument lub `NOCARZ_MODE`):

| Tryb | Co robi |
|---|---|
| `serve` *(domyślny)* | uruchamia mikroserwis na `0.0.0.0:8080` |
| `pipeline` | buduje tabele cech/celu i trenuje + zapisuje cztery modele (przychód + obłożenie, A + B) |
| `ab` | uruchamia pełny eksperyment A/B (serwer → odtworzenie ofert → ewaluacja) |
| `test` | uruchamia testy (`pytest`) |

```bash
# 1) Potok offline: cel → cechy → trening → ground truth.
#    (krok calendar→cel ~33 mln wierszy jest pomijany, jeśli artefakt istnieje;
#     wymuszenie pełnego przeliczenia: -e NOCARZ_FORCE=1)
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" nocarz pipeline

# 2) Mikroserwis (http://127.0.0.1:8080)
docker run --rm -p 8080:8080 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" nocarz

# 3) Eksperyment A/B (sam startuje serwer w tle, odtwarza oferty, liczy metryki)
docker run --rm -v "$PWD:/app" -e NOCARZ_AB_N=1000 nocarz ab

# 4) Testy
docker run --rm -v "$PWD/data:/app/data" nocarz test
```

Wolumeny: `data/` montujemy zawsze (serwis czyta `listings.csv` i pliki w `data/processed/`).
`models/` montujemy, gdy chcemy, by `pipeline` zapisał świeże modele na hoście; do samego `serve`
nie jest wymagany (modele są wbudowane w obraz). Tryb `ab` zapisuje raport, więc montujemy całe repo.

## Użycie API

Odpowiedź **nie ujawnia**, który model odpowiedział — wybór A/B jest przezroczysty dla klienta.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"features":{"listing_id":3109,"latitude":48.8319,"longitude":2.3187,
       "neighbourhood_cleansed":"Observatoire","property_type":"Entire rental unit",
       "room_type":"Entire home/apt","accommodates":2,"amenities_count":15}}' \
  http://127.0.0.1:8080/predict_revenue
# -> {"request_id": "...", "listing_id": 3109, "predicted_annual_revenue": 36683.32,
#     "predicted_occupancy": 0.62, "currency": "EUR"}
```

Pola `client_id`, `force_model` oraz `bathrooms` i `premium_amenities_count` są **opcjonalne**
(brakujące cechy lokalu są imputowane medianą/zerem), więc minimalny payload zawiera tylko
`features` z podstawowymi atrybutami.

Pozostałe endpointy:
- `GET /health` — żywotność + wersje załadowanych modeli.
- `POST /predict_revenue/{a|b}` — wymuszenie modelu (ujawnia jego tożsamość); używane przez testy
  i parowaną analizę A/B.

Każde żądanie jest logowane (`logs/predictions.jsonl`): przypisany model i jego wersja, cechy
wejściowe i pochodne, predykcja oraz latencja — to wejście dla ewaluacji A/B.

## Eksperyment A/B

Tryb `ab` (wyżej) wykonuje całość. Pod spodem: `simulate_clients.py` odtwarza wydzielone oferty
testowe (nieużyte w treningu) przez serwis, a `evaluate_ab.py` łączy log z `ground_truth.csv`
i raportuje metryki per model, istotność statystyczną (Mann‑Whitney/Welch na ruchu niezależnym,
**parowany Wilcoxon** na wymuszonych `/a`–`/b` — najmocniejszy sygnał) oraz werdykt
do `reports/ab_report.md`.

## Notatniki i raport

`notebooks/01_eda`, `02_modeling` (mapa potencjału + „białe plamy"), `03_ab_evaluation` to wykonane
artefakty raportowe (po polsku). Logika `02`/`03` współdzieli funkcje z `scripts/`. Dla edycji
notatników w Jupyterze użyj środowiska Python 3.12.

## Uwagi

- **Spójna wersja Pythona:** cały stack (obraz Docker, zapisane modele) jest na **3.12**.
- **`127.0.0.1` zamiast `localhost`:** symulator używa `127.0.0.1`, by uniknąć wolnego fallbacku IPv6.
- **Kodowanie:** dane są w UTF‑8 (np. „Élysée"); polskie/francuskie znaki bywają zniekształcone
  jedynie w konsoli cp1252 — w plikach są poprawne.
- **Praca bez Dockera (dev):** scenariusze deweloperskie da się uruchomić lokalnie pod Pythonem 3.12
  (`PYTHONPATH=src`), ale wspieraną i powtarzalną ścieżką uruchomienia jest Docker (powyżej).
