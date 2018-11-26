#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:21:24 2018

@author: iris
"""


from Traitement_fich import*

time=50

chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/' #à changer selon les ordinateurs

T=[round(i/512,6) for i in range(1,time*512+1)]

#T=open_data(chemin+'BP1-BP2_Temps.txt')[0:time*512]
#print(T[1:10])
##Afin de pouvoir étudier les motifs sur un nouvel alogrithme, on veut avoir 

#On extrait la série temporelle brut : #
#Pour B
#YB=open_data(chemin+"B3-B2N3.txt")[0:time*512]
#YO=open_data(chemin+"O'9-O'8N3.txt")[0:time*512]

#On extrait la série en puissance il faut qu'elle soit de la même longueur
#PB=calc_puiss(chemin+"B3-B2N3.txt",T,h=1,opt='ripples')[0] #on choisit de faire un pas de 1
#PO=calc_puiss(chemin+"O'9-O'8N3.txt",T,h=1,opt='delta')[0]
#aff_puiss(chemin+"B3-B2N3.txt",T,h=5,opt='ripples')

#Construction D'un vecteur de 0 et de 1 correspondant à l'apparition de sharpw

#vect_detect_pic(chemin+"B3-B2N3.txt",T)

##Detection des pics delta de la même manière que l'on a détecté les sharp waves ripples
#detect_delta(chemin+"O'9-O'8N3.txt",T,1)

#Construction d'un vecteur de 0 et de 1 caractérisant la présence de pic delta
Y=vect_detect_pic(chemin+"O'9-O'8N3.txt",T,'delta',2,100)

#vecteur pour la phase : 
#phi=phase_delta(chemin+"B3-B2N3.txt",chemin+"O'9-O'8N3.txt",T,1)