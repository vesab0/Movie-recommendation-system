Spjegim i arkitektures se projektit 

### Faza 1 

Faza 1 perfshin pergaditjen e te dhenave, krijimin e vektoreve dhe aftesine e sistemit per te gjeneruar rekomandime ne baze te kontentit te filmit input

##### Hapi 1: 

**Loader.py**

Loader.py merret me krijimin e strukturave te python per te filluar algoritmin. Te dhenat jane ne formen .csv (excel files). Loader.py lexon ato dhe i ndan ne 3 struktura:
1. valid_proviles - dictionary i filmave me te gjitha te dhenat e tyre 
2. ratings_by_movie - kthen te gjitha user ratings per filmin 
3. ratings_by_user - kthen te gjitha ratings te nje user

##### Hapi 2: 

**vectorizer.py**

Per te funksionuar algoritmi KNN, te gjitha te dhenat duhet te kthehen ne numra dhe te gjitha features ne dimenzione. Kjo esht puna e vectorizer, i cili merr cdo film edhe e kthen ne vektor te numrave qe perfaqson nje pik ne hapsir N (35 ne datasetin tone) dimenzionale. 

Per perkthimin e vlerave string ne vlera numerike, vectorizer.py krijon dictionaries ku cdo vlere unike qe gjindet ne dataset i behet assign ni value. Largohen vlerat te cilat nuk plotesojne nje kusht te caktuar, qe te mos ngarkohet modeli me psh aktor qe jan vetem ne 1 film 2, apo vlera tjera te cilat nuk perseriten mjaftueshem qe me ndiku ne llogaritje te ngjashmerise.

Vlerat boolean vazhdojne si 1 ose 0 kurse vlerat numerike mbesin te tilla

Merren filmat e fituar nga Hapi 1 dhe per secilin cdo feature kthehet ne vlere numerike.

**cacher.py**

Cacher.py thirr loader.py dhe vectorizer.py in that order dhe i ben cache rezultatet.

Perdoret libraria numpy qe ofron array object me veprime 50x me te shpejta se array klasik i python, si dhe operacione tjera matematikore me te shpejta. Ofron kompresim te strukturave numerike, na lejon qe vector cache te ket madhesi te arsyeshme

##### Hapi 3: 

**content_knn.py**

content_knn.py llogarit distancen e pikes qe na zgjedhim (filmit ne ket rast), me cdo pik tjeter ne graf. I kthen K (10) te paret 

**collaborative_knn**

content_knn.py loops through every movie ne list, nese filmi ka common reviewers me filmin input, ather e bon run formulen e cila e kontrollon se users qe e bojn rate mire filmin input a e bojn rate mire edhe filmin tjeter. Nfund ata filma best rated prej common reviewers dalin 

