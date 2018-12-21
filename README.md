# Projet_Correlation_sharpw_delta

Les fichiers EEG extraits ont une frequence d'échantillonage de 512 Hz

## Traitement_fich
Fichier qui contient toutes les fonctions utilisées pour l'analyse des signaux.
#### filtre : 
Filtre les données issus des fichiers mis en entrées. Si il s'agit de signaux venant des electrodes B (dans l'hippocampe), il faut préciser l'option ripples et ainsi le filtrage se fera entre 120 et 250 hz. Pour des signaux provenant des zone où l'on s'attend à observer des rythmes delta, le filtrage se fait entre 1 et 4 Hz

#### calc_puiss : 
Fonction qui calcule retourne la puissance du signal en entrée. Il y a un pas (h) qui peut être choisi, par défaut ce pas est de 20 points soit 39 ms. Si l'option est delta la fenêtre de calcul de la puissance est plus élevé (environ 1 seconde). Tandis que pour les signaux issus des electrodes B la fenêtre de calcul est de 78 ms. 

#### detec_pic : 
Il s'agit d'une fonction qui determine les temps de debut, fin et maximum d'un sharp waves ripples. 
Nous nous sommmes enuite intéressé à la supression des pics epileptique. 
Du coup si l'option 'ripples' est utilisé nous vérifions qu'à 500 ms près cela ne coincide pas avec l'apparition d'un pic epileptique détecté au niveau des elctrodes A'2-A'1.Il s'agit de la fonction clean_epileptic_pic
Il reste cependant une erreu un peu étonnante vis à vis des valeurs maximal de sharpw, mais je devrais réussir à le corriger bientôt. 

#### sort_sharpw_ripples : 
Fonction qui affiche sur le graphe du signal puissance les sharps waves de différentes couleurs suivant la valeur de leur amplitude. 
#### Repartition de la phase : phase_delta et stat_phase
Nous étudions la répartition de la phase en fonction de l'amplitude des sharp waves. 
## Traitement_sig
Ce fichier est utilisé pour appliqué les fonctions de Traitement fich aux différents fichiers de notre choix. 
## Creation_serieT 
Fichier qui fait appel aux fonctions de Traitement_fich afin de créer des séries temporelles qui seront analysé par DTW. (Dynamic Time Warping). Puisqu'il faut que les serie temporelles soient de même longueur, on choisit de calculer la puissance avec un pas de 1. Si il y a d'eventuel probleme lié à ce pas très faible il est envisageable de garder lun pas de 20 points pour la puissance et de faire une interpolation pour rajouter les points manquants. 

## Améliorations 

Il faut penser à optimiser le code, pour permettre travailler sur un grand jeux de données. 
