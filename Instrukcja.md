# Projekt IUM – 2026L

# v.1.

## Kontekst

W ramach projektu wcielamy się w rolę analityka pracującego dla portalu Nocarz – serwisu, w
którym klienci mogą wyszukać i dokonać rezerwacji noclegu (zapewnianego przez oferentów).
Praca na tym stanowisku nie jest łatwa – zadanie dostajemy w formie enigmatycznego opisu i to
do nas należy doprecyzowanie szczegółów tak, aby dało się je zrealizować. To oczywiście wymaga
zrozumienia problemu, przeanalizowania danych, czasami negocjacji z szefostwem. Poza tym,
oprócz przeanalizowania zagadnienia i wytrenowania modeli, musimy przygotować je do
wdrożenia produkcyjnego –zakładając, że w przyszłości będą pojawiać się kolejne ich wersje, z
którymi będziemy eksperymentować.

Jak każda szanująca się firma internetowa, Nocarz zbiera dane dotyczące swojej działalności – są
to (analitycy mogą wnioskować o dostęp do tych informacji na potrzeby realizacji zadania):
● szczegółowe dane o dostępnych lokalach,
● recenzje lokali,
● kalendarz z dostępnością i cenami,
● baza klientów i sesji.

## Zadanie

Prowadzący: Dariusz Jagodziński dariusz.jagodzinski@pw.edu.pl

12. "Czasami zastanawiamy się w którym miejscu powinniśmy poszukać jakiegoś nowego
    lokalu do dodania, tak aby był jak najbardziej opłacalny"

## Wspólne zasady

1) Projekt **realizujemy w parach** – każda para otrzymuje do realizacji jedno z zadań z listy
powyżej.

2) W trakcie trwania projektu prowadzący pełni rolę klienta, dla którego realizujecie Państwo
zadanie biznesowe.

3) **Zgłoszenia do zadań realizujemy poprzez aplikację** :

```
a) adres ( dostępny jedynie z sieci PW ): http://assigner.ii.pw.edu.pl/
b) logowanie – jak do USOS, potem wybranie przedmiotu IUM,
c) proszę o wybieranie 2-3 preferowanych zadań – zapisy na zadania trwają do 2026.03.29.
```
4) W ramach projektu trzeba dostarczyć:

```
a) etap 1 – możliwe, że będzie konieczne wykonanie więcej niż jednej iteracji tego etapu –
pierwsza powinna zakończyć się do 2026.05.03 (0-20 pkt ):
i) wypełniony dokument IUM – Machine Learning Canvas (v1.0)^1 (zawiera definicję
problemu biznesowego, zdefiniowanie zadania/zadań modelowania, założenia,
zaproponowame kryteria sukcesu),
ii) analizę danych z perspektywy realizacji tych zadań (trzeba ocenić, czy dostarczone
dane są wystarczające – może czegoś brakuje, może coś trzeba poprawić, domagać się
innych danych, ...),
b) etap 2 – do 2026.06.07 :
i) dwa modele: model bazowy (najprostszy możliwy dla danego zadania) i bardziej
zaawansowany model docelowy, oraz raport z pokazujący proces budowy modelu i
porównujący wyniki (0-15 pkt) ,
```
(^1) Jest to zaadaptowany na potrzeby projektu dokument zgodny z procesem Machine Learning Canvas – opisanym
tutaj.


```
ii) implementację mikroserwisu (przykład czym jest mikroserwis można zobaczyć tutaj),
która pozwala na (0-15 pkt) :
(1) serwowanie predykcji – tutaj proszę pamiętać, że wybór modelu z perspektywy
klienta Państwa usługi powinien być przezroczysty, a przykładowe wywołanie
mogłoby wyglądać choćby tak:
```
```
curl -X POST -H "Content-Type:application/json" -d '{json data goes here}'
http://localhost:8080/some_app_name
```
(2) realizację eksperymentu A/B – w ramach którego porównywane będą oba modele i
**zbierane dane** niezbędne do późniejszej oceny ich jakości,
iii) skrypt/notatnik pozwalający na ewaluację testu A/B na podstawie logu generowanego
przez mikroserwis
iv) materiały pokazujące, że implementacja działa.
5) “Oddając” etap, wysyłamy rozwiązanie mailem do Prowadzącego w tym:

```
a) raporty/dokumentację w formie PDF lub notatników jupyter (ułatwiających łączenie
generowanych wyników/wykresów i tekstu raportu w jednym dokumencie),
b) implementację (etap 2) – w pliku zip (proszę nie wysyłać formatu rar – serwer PW często
oznacza takie wiadomości jako spam).
```
6) Zachęcam do korzystania z **konsultacji** w przypadku jakichkolwiek wątpliwości – najpóźniej
do **2026.05.31 –** ostatni tydzień przed terminem traktujemy jako „finalny sprint”, podczas
którego wszystkie ustalenia są już zamrożone.
7) Niedotrzymanie terminów (czyli niewysłanie rozwiązania do prowadzącego – konsultacje
mogą być później) skutkuje karą za **spóźnienia: -2 pkt za każdą rozpoczętą dobę**.



