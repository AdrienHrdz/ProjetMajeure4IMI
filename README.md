# **Projet Majeure**

### Sujet : 
Detection de grains de beautés cancéreux.
Pour cela on utilise les 5 critères suivants :


## A : asymetrie
Forme ni ronde ni ovale.
Mesure de similarité par rotation de la forme.
Snake pour avoir les bords de l'object.
4 points pour trouver le centre.
Rotation de la forme depuis ce cercle.
Mesure du nombre de pixels de différences.

## B : bords
Bords mal délimités, irréguliers \
Travail sur une coupe de l'image. 
TV pour lisser la texture 
detection de la valeur de la pente pour avoir le score. Pour mettre en oeuvre cela on utilise le code `Matlab` du TP2 variation totale.
Problème avec l'utilisation de python, on reste sur `Matlab`. 
Après TV, on a une image lissée sans texture. On calcule ensuite le gradient de cette image afin d'obtenir une information sur les pentes au niveau des contours.
On effectue un moyennage de la norme du gradient au carré sur le nombre de pixels non nuls. 
Ainsi on peut obtenir un score sur l'irregularité des bords.


## C : couleur
Présence de plusieurs couleurs 

## D : diametre
Anormal si supérieur à 6mm \
Mesure par morphoMath \
Utilisation de la librairie `cv2.aruco` pour la detection de marqueur et recalage de l'image. 

**TODO** : refaire l'image avec les marqueurs et diminuer la distance entre les marqueurs. 

## E : evolution
Changement de topologie 
