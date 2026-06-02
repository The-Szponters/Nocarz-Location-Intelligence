# Nocarz — modele opłacalności lokalizacji + mikroserwis z A/B

Projekt IUM (temat 12). Przewidujemy **roczny przychód** lokalu na podstawie cech znanych
przed wystawieniem oferty, aby wskazać działowi Business Development najbardziej opłacalne
lokalizacje („białe plamy"). Całość: dwa modele + raport, mikroserwis serwujący predykcje
z przezroczystym wyborem modelu i eksperymentem A/B, oraz ewaluacja A/B z loga.

Pełny opis metodyki i wyników: **`reports/raport.md`**.

## Struktura

```
src/nocarz/      kod współdzielony (cechy, schematy, routing, logowanie, rejestr, FastAPI)
scripts/         potok offline + serwer + symulacja + ewaluacja
notebooks/       01_eda · 02_modeling (mapa „białych plam") · 03_ab_evaluation  (po polsku)
models/          zapisane modele + registry.json
data/processed/  tabele pośrednie (cel, cechy, zbiór testowy, ground truth)
reports/         raport.md, ab_report.md, figures/
tests/           testy (pytest)
```

## Wymagania

Python **3.12** (cały stack — venv, Docker, modele — jest na 3.12, patrz Uwagi). Instalacja zależności:

```powershell
python -m pip install -r requirements.txt
```

Jedyny pakiet wymagający doinstalowania ponad standardowe środowisko to `uvicorn`.

## Potok danych i modeli (kolejność)

```powershell
python scripts/build_targets.py      # calendar.csv (33 mln) -> cel (przychód), strumieniowo
python scripts/build_features.py     # listings + cechy przestrzenne -> model_table.csv
python scripts/train_models.py       # baseline + HGB, CV przestrzenna (LODO), zapis modeli + test_set
python scripts/make_ground_truth.py  # ground_truth.csv (id -> prawdziwy przychód, zbiór testowy)
```

## Docker (zalecane przy ewaluacji — cross-platform)

Obraz oparty na **Python 3.12** daje spójne, powtarzalne środowisko (te same wersje
co lokalny venv, więc brak problemu `PCG64 is not a known BitGenerator`) oraz omija
wolny fallback IPv6 `localhost` na hoście.

```bash
docker build -t nocarz .
# data/ i models/ są w .gitignore — podmontuj je (albo zostaną wbudowane przez COPY, jeśli są obecne):
docker run --rm -p 8080:8080 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  nocarz
# serwis nasłuchuje na http://127.0.0.1:8080  (GET /health, POST /predict_revenue)
```

W kontenerze uvicorn słucha na `0.0.0.0:8080`; mapowanie `-p 8080:8080` udostępnia go na hoście.

## Mikroserwis

PowerShell (Windows):

```powershell
.\scripts\run_server.ps1             # uvicorn na http://127.0.0.1:8080
```

bash (Linux / macOS):

```bash
PYTHONPATH=src python -m uvicorn nocarz.app:app --host 127.0.0.1 --port 8080 --workers 1
```

Przykładowe wywołanie (zgodne z poleceniem). W PowerShell `curl` to alias `Invoke-WebRequest`,
dlatego używamy **`curl.exe`** albo formy natywnej:

```powershell
# curl.exe (dosłownie jak w poleceniu)
curl.exe -X POST -H "Content-Type: application/json" `
  -d '{\"features\":{\"listing_id\":3109,\"latitude\":48.8319,\"longitude\":2.3187,\"neighbourhood_cleansed\":\"Observatoire\",\"property_type\":\"Entire rental unit\",\"room_type\":\"Entire home/apt\",\"accommodates\":2,\"amenities_count\":15}}' `
  http://localhost:8080/predict_revenue

# forma natywna PowerShell (bez problemów z cudzysłowami)
$body = @{ features = @{ listing_id=3109; latitude=48.8319; longitude=2.3187;
  neighbourhood_cleansed="Observatoire"; property_type="Entire rental unit";
  room_type="Entire home/apt"; accommodates=2; amenities_count=15 } } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/predict_revenue -Method Post `
  -ContentType "application/json" -Body $body
```

bash (Linux / macOS) — `client_id` i `force_model` są opcjonalne, więc wystarczy `features`:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"features":{"listing_id":3109,"latitude":48.8319,"longitude":2.3187,"neighbourhood_cleansed":"Observatoire","property_type":"Entire rental unit","room_type":"Entire home/apt","accommodates":2,"amenities_count":15}}' \
  http://127.0.0.1:8080/predict_revenue
```

Odpowiedź **nie ujawnia** użytego modelu (wybór A/B przezroczysty dla klienta).
Endpointy pomocnicze: `GET /health`, `POST /predict_revenue/{a|b}` (wymuszenie modelu).

## Eksperyment A/B

```powershell
python scripts/simulate_clients.py --n 1000 --paired   # odtworzenie ofert testowych -> log
python scripts/evaluate_ab.py                          # metryki + istotność + werdykt -> reports/ab_report.md
```

## Notatniki

```powershell
python -m ipykernel install --user --name nocarz-py312 --display-name "Python 3.12 (nocarz)"
python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=nocarz-py312 notebooks/*.ipynb
```

## Testy

PowerShell (Windows):

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/ -q
```

bash (Linux / macOS):

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

## Uwagi (Windows / PowerShell)

- **Spójna wersja Pythona:** cały stack (venv, obraz Docker, zapisane modele) jest na **3.12**.
  Modele trenuj i serwuj tym samym interpreterem — mieszanie wersji minor (np. 3.12 vs 3.13)
  psuje odczyt modeli (`PCG64 is not a known BitGenerator`). Dla notatników rejestrujemy i
  używamy kernela 3.12 (komenda wyżej).
- **`localhost` vs `127.0.0.1`:** klient Pythonowy (`urllib`) bywa wolny na `localhost`
  (fallback IPv6). W symulatorze domyślnie używamy `127.0.0.1`.
- **Kodowanie:** dane są poprawnym UTF‑8 (np. „Élysée"); polskie/francuskie znaki mogą wyglądać
  na zniekształcone tylko w konsoli cp1252 — w plikach są poprawne.
