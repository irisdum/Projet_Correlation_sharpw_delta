#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:28:31 2018

@author: iris
"""

from Traitement_fich import*

time=50

chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/' #à changer selon les ordinateurs

T=open_data(chemin+'BP1-BP2_Temps.txt')[0:time*512] 

#T=[i/512 for i in range(1,time*512+1)]

##Affichons les signaux bruts

#trace(chemin+"O'9-O'8N3.txt",T)


##Affichons les corrélations entre deux signaux : 
#nom1="B3-B2N3.txt"
#nom2="B4-B3N3.txt"
#Y1=open_data(chemin+"B3-B2N3.txt")[0:20*512]
#Y2=open_data(chemin+"B4-B3N3.txt")[0:20*512]
#print(compare(Y1,Y2,nom1[:-4]+nom2[:-4],'brut'))

##Affichons les signaux filtrés et la puissance calculée

#aff_puiss(chemin+"O'9-O'8N3.txt",T,h=20,opt='delta')


##Compare les signaux filtrées venant de l'hippocampe et electrode O9-O8:
#compar_delta_ripples(T,chemin+"B3-B2N3.txt",chemin+"O'9-O'8N3.txt")

#Correlation en puissance entre electrode de l'hippocampe

#corr_sig_puiss(chemin+"B'2-B'1N3.txt",chemin+"B'4-B'3N3.txt",T)  #penser à modifier un bout du code. enlever l'option delta

##Correlation en puissance : Calcule la correlation entre deux signaux. Le premier contient des sharpwaves l'autre pour les delta
#filtre
#corr_sig_puiss(chemin+"B'2-B'1N3.txt",chemin+"O'9-O'8N3.txt",T) 

## Eventuelle périodicité entre les signaux bruts O'9-O'8 et B'2-B'1
##Calcule de cette période : on se place soit dans le cas de décalage positif ou négatif 
#print(np.mean(periode_corr(chemin+"B'2-B'1N3.txt",chemin+"O'9-O'8N3.txt",0.10,1,0)))

##Etudions si les sharpw sont détécté à peu près au même moment pour deux electrodes proches dans l'hippocampe
#detec_pic(chemin+"B'4-B'3N3.txt",T,3,5,1)
#ddetec_pic(chemin+"B'2-B'1N3.txt",T)
#ddetec_pic(chemin+"B'3-B'2N3.txt",T)
#Etudions les differents type de sharpw soit trié en fonction de leur écart à la moyenne
#sort_sharpw_ripples(chemin+"B3-B2N3.txt",'ripples',1)

##Etudions si des sharps waves rippples coincident graphiquement avec l'apparition de delta

#detec_delta_sharp_ripples(chemin+"O'9-O'8N3.txt",chemin+"B'2-B'1N3.txt")

##Verifions que les pics observé correspondent bien à des sharpw et non des pics epileptique

#detec_epileptic_pic(chemin+ "A'2-A'1N3.txt",chemin+"B'4-B'3N3.txt",T)

#récupérons la phase du signal lorsqu'un sharpw est détecté
#phase_delta(chemin+"B'2-B'1N3.txt",chemin+"O'9-O'8N3.txt",T,20)


