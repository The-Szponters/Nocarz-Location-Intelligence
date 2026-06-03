# IUM MACHINE LEARNING CANVAS Nr tematu: 12 Zespół: The Szponters Data: 28.04.2026 Iteracja: 1.

### ZADANIA MODELOWANIA

```
Typ zadania:
Przestrzenna regresja punktowa połączona
z logiką decyzyjną (post-processing).
Algorytm ML przewiduje ciągłe wartości
finansowe i wskaźnikowe. Ranking
lokalizacji jest generowany poza modelem.
Model nie ogranicza się do oceny
istniejących ofert, lecz generuje predykcje
dla syntetycznej siatki geograficznej
nałożonej na analizowany region.
Dane wejściowe:
Koordynaty geograficzne, atrybuty
typologiczne (typ nieruchomości, pokoju),
parametry fizyczne lokalu oraz dane
makroprzestrzenne.
Możliwe wyniki (Predykcja):
Szacowany przychód roczny dla profilu
referencyjnego oraz przewidywane
obłożenie.
Obserwacja etykiet:
Wartości Ground Truth są obserwowane po
```
## 365 dniach od aktywacji lokalu w systemie.

### DECYZJE

```
Surowe predykcje modelu są
transformowane w konkretne działania
biznesowe:
```
**1.** Dla każdej komórki siatki obliczany jest
Wskaźnik Potencjału Przychodowego (WPP)
**2.** Spatial Join wyznacza Indeks Nasycenia
(Market Saturation) zliczając aktywne oferty
w promieniu 500m.
**3.** Jeśli spełnione poniższe warunki to
obszar otrzymuje status „White Spot” (biała
plama z wysokim potencjałem).
    a) WPP > Próg_Rentowności
    b) Indeks Nasycenia < Próg_Saturacji
**4.** System generuje listę rankingową
"Top-N" lokalizacji dla działu Business
Development.

### PROPOZYCJA WARTOŚCI

```
Beneficjenci:
Dział Rozwoju Biznesu: Mniej
straconego czasu na pozyskiwanie
nierentowych ofert. Precyzyjne
uderzenie marketingowe tam, gdzie
zwrot będzie najwyższy i najszybszy.
Menedżerowie Produktu: Uniknięcie
kanibalizacji ofert (Market Saturation) na
jednym obszarze, co zapobiega wojnom
cenowym i chroni prowizje serwisowe.
Zarząd Nocarz: Decyzje o ekspansji (np.
wejście do nowej dzielnicy) poparte
modelem Geospatial AI, a nie
przeczuciem.
Zintegrowany system uczenia
maszynowego:
Interaktywny widok mapy ciepła dla
analityków oraz bezpośrednie
```
## generowanie list leadów w systemie CRM

## ZBIERANIE DANYCH

```
System wymaga ciągłej
aktualizacji wiedzy o rynku poprzez pętlę
zwrotną (feedback loop).
Weryfikacja realności: Aktualne wahania
cen i dostępności z calendar.csv służą do
kalibracji bazowych wyników dzielnicy.
Analiza popytu: Dane z sesji oraz recenzje
opierają się na komentarzach i dacie.
Wczesne zainteresowanie rejonem badane
jest za pomocą agregacji sentymentu
(NLP) oraz dynamiki przyrostu
komentarzy.
Ewolucja trendów: Dane z recenzji są
agregowane przestrzennie, aby wykrywać
zmiany w postrzeganiu atrakcyjności
dzielnic przez gości.
```
## ŹRÓDŁA DANYCH

```
Zbiór uczący budowany jest w
oparciu o ekstrakcję cech z następujących
struktur:
Cechy egzogeniczne:
latitude, longitude,
neighbourhood_cleansed.
Specyfikacja lokalu:
property_type, room_type,
accommodates, bathrooms_text
(wymagający parsowania tekstu na
wartości numeryczne) oraz amenities
(liczba udogodnień i obecność cech
premium).
Etykiety (Ground Truth):
estimated_revenue_l365d oraz
estimated_occupancy_l365d.
Zmienne kontrolne:
review_scores_location (jako proxy dla
atrakcyjności punktowej) oraz historyczna
zmienność ceny z calendar_prev.csv.
```
### WPŁYW PROJEKTU

```
Koszty błędów:
```
- Błędna rekomendacja: Koszt
    pozyskania oferenta w lokalizacji, która
    nie osiąga progu rentowności. Skutkuje
    to niezadowoleniem oferentów i
    stratami operacyjnymi.
- Pominięcie "żyły złota": Koszt
    utraconych korzyści na rzecz
    konkurencji w regionie
**Kryteria sukcesu**
- Osiągnięcie RMSE dla prognozy
rocznego przychodu poniżej 12%
odchylenia standardowego w danym
klastrze geograficznym.

## GENEROWANIE PREDYKCJI

```
Tryb Wsadowy (Batch): Raz w
tygodniu generowanie pełnej mapy
potencjału dla wszystkich regionów
operacyjnych.
Tryb On-demand: Natychmiastowe
generowanie predykcji (< 1s) w panelu
analitycznym.
Uzasadnienie: Proces szukania lokalu
wymaga stabilnych danych trendowych
(Batch), ale w fazie weryfikacji konkretnych
adresów niezbędna jest natychmiastowa
```
## reakcja systemu (On-demand).

### BUDOWANIE MODELI

```
Architektura: Jeden ujednolicony model
oparty na algorytmach Gradient Boosting
przystosowanych do danych
stabelaryzowanych, integrujący
przestrzenne złączenia.
Walidacja: Ze względu na autokorelację
przestrzenną, walidacja odbywa się
wyłącznie metodą Spatial
Cross-Validation (np.
Leave-One-District-Out). Zwykły random
split jest zabroniony (Data Leakage).
Cykl życia: Douczanie modelu co
miesiąc przy użyciu zaktualizowanego
snapshota bazy ofert.
Czas: Agregacje przestrzenne zajmą do
120 roboczogodzin maszyny,
optymalizacja hiperparametrów do 6
```
## godzin.

### ATRYBUTY

```
Atrybuty Geograficzne:
Współrzędne wektorowe, kodowanie
dzielnicy, odległość od kluczowych
punktów.
Atrybuty Otoczenia : Gęstość konkurencji
w promieniu 250m, 500m i 1000m,
średnia ocena lokacji w sąsiedztwie,
mediana ceny w najbliższym klastrze.
Atrybuty Historyczne: zmienność ceny w
kalendarzu, historyczna dynamika zmian
```
## obłożenia w regionie.

- Współczynnik konwersji pozyskanych
    lokali w rekomendowanych strefach
    wyższy o 20% w porównaniu do grupy
    kontrolnej.

## MONITOROWANIE

```
Dla inżynierów ML: Błąd predykcyjny na danych produkcyjnych (RMSE, MAE). Automatyczne narzędzia śledzące Dryf Danych
na dystrybucji kolumn cen i przewidywanych etykietach, aby wykryć zmiany ekonomiczne.
Dla biznesu: Delta przychodu (Revenue Lift) lokali z polecenia vs lokale organiczne, udział (Market Share) w strefach "White Spot"
Harmonogram przeglądów: Cotygodniowy raport techniczny metryk modelu (RMSE, MAE); miesięczna rewizja biznesowa strategii
ekspansji przestrzennej.
Atrybuty Typologii i Standardu : Lista
udogodnień, liczba miejsc noclegowych.
Dokument jest utworzony na podstawie: OWNML MACHINE LEARNING CANVAS v 1.2. Created by Louis Dorard, Ph.D. Licensed under a Creative
```
Commons Attribution-ShareAlike 4.0 International License. (^) **OWNML.CO**



