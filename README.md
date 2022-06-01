# **Projet Majeure**

### Sujet : 
Detection de grains de beautés cancéreux.
Pour cela on utilise les 5 critères suivants :


## A : asymetrie
Forme ni ronde ni ovale 

## B : bords
Bords mal délimités, irréguliers \
Travail sur une coupe de l'image. 
TV pour lisser la texture 
detection de la valeur de la pente pour avoir le score. Pour mettre en oeuvre cela on utilise le code `Matlab` du TP2 variation totale.

## C : couleur
Présence de plusieurs couleurs 

## D : diametre
Anormal si supérieur à 6mm \
Mesure par morphoMath \
Utilisation de la librairie `cv2.aruco` pour la detection de marqueur et recalage de l'image. 

**TODO** : refaire l'image avec les marqueurs et diminuer la distance entre les marqueurs. 

## E : evolution
Changement de topologie 
