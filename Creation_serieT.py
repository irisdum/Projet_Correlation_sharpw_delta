#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:21:24 2018

@author: iris
"""


from Traitement_fich import*

time=20 #Au debut

chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/' #à changer selon les ordinateurs

T=[round(i/512,6) for i in range(1,time*512+1)]

charB=chemin+"B3-B2N3_60s.txt"
charO=chemin+"O'9-O'8N3_60s.txt"
##Afin de pouvoir étudier les motifs sur un nouvel alogrithme, on veut avoir 

#On extrait la série temporelle brut : #
#Pour B
#YB=open_data(charB)[0:time*512]
#YO=open_data(charO)[0:time*512]

#On extrait la série en puissance il faut qu'elle soit de la même longueur
PB=calc_puiss(charB,T,h=1,opt='ripples')[0] #on choisit de faire un pas de 1
PO=calc_puiss(charO,T,h=1,opt='delta')[0]
#aff_puiss(chemin+charB,T,h=5,opt='ripples')

fich1=open("puiss"+charB[66:-4]+".txt","w")
fich1.write('Puissance du signal\n')
fich1.write(charB[66:-4]+'\n')
for elem in PB : 
    fich1.write(str(elem)+'\n')
fich1.close()

fich2=open("puiss"+charO[66:-4]+".txt","w")
fich2.write('Puissance du signal\n')
fich2.write(charO[66:-4]+'\n')
for elem in PO : 
    fich2.write(str(elem)+'\n')
fich2.close()

#Construction D'un vecteur de 0 et de 1 correspondant à l'apparition de sharpw

VB=vect_detect_pic(charB,T,'ripples',3,15)
print(len(VB))
fich3=open("Vect_pic"+charB[66:-4]+".txt","w")
fich3.write('Detection de pic en amplitude \n')
fich3.write(charB[66:-4]+'\n')
for elem in VB : 
    fich3.write(str(elem)+'\n')
fich3.close()
#Construction d'un vecteur de 0 et de 1 caractérisant la présence de pic delta
##Detection des pics delta de la même manière que l'on a détecté les sharp waves ripples
VO=vect_detect_pic(charO,T,'delta',2,100)
fich4=open("Vect_pic"+charO[66:-4]+".txt","w")
fich4.write('Detection de pic en amplitude \n')
fich4.write(charO[66:-4]+'\n')
for elem in VO : 
    fich4.write(str(elem)+'\n')
fich4.close()
#Construction d'un vecteur de 0 et de 1 caractérisant la présence de pic delta


#vecteur pour la phase : 
phi=phase_delta(charB,charO,T,1)
fich5=open("Phase_"+charO[66:-4]+".txt","w")
fich5.write('Phase \n')
fich5.write(charO[66:-4]+'\n')
for elem in phi : 
    fich5.write(str(elem)+'\n')
fich5.close()
