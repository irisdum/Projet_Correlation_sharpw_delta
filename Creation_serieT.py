#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:21:24 2018

@author: iris
"""

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from Traitement_fich import*


time=20 #Au debut

chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/' #à changer selon les ordinateurs
size= time*512 #taille des listes
T=[round(i/512,6) for i in range(1,time*512+1)]

charB=chemin+"B3-B2N3_120s.txt"
charO=chemin+"O'9-O'8_120s.txt"
charA=chemin+"A'2-A'1_120s.txt"
##Afin de pouvoir étudier les motifs sur un nouvel alogrithme, on veut avoir 

#On extrait la série temporelle brut : #
#Pour B
YB=open_data(charB)[0:time*512]
YO=open_data(charO)[0:time*512]

#On extrait la série en puissance il faut qu'elle soit de la même longueur
#PB=calc_puiss(charB,T,h=1,opt='ripples')[0] #on choisit de faire un pas de 1
#PO=calc_puiss(charO,T,h=1,opt='delta')[0]
PB=normalise_puiss(charB,T,1,'ripples')
PO=normalise_puiss(charO,T,1,'delta')

#print("pour PO",len(PO))
#print("pour PB",len(PB))
#aff_puiss(chemin+charB,T,h=5,opt='ripples')

#fich1=open("puiss"+charB[66:-4]+".txt","w")
#fich1.write('Puissance du signal\n')
#fich1.write(charB[66:-4]+'\n')
#for elem in PB : 
#    #fich1.write(str(elem)+'\n')
#fich1.close()
#
#fich2=open("puiss"+charO[66:-4]+".txt","w")
#fich2.write('Puissance du signal\n')
#fich2.write(charO[66:-4]+'\n')
#for elem in PO : 
#    #fich2.write(str(elem)+'\n')
#fich2.close()

#Construction D'un vecteur de 0 et de 1 correspondant à l'apparition de sharpw

#VB=vect_detect_pic(charB,charA,T,'ripples',3,15) 
#print(len(VB))
#fich3=open("Vect_pic"+charB[66:-4]+".txt","w")
#fich3.write('Detection de pic en amplitude \n')
#fich3.write(charB[66:-4]+'\n')
#for elem in VB : 
#    #fich3.write(str(elem)+'\n')
#fich3.close()
##Construction d'un vecteur de 0 et de 1 caractérisant la présence de pic delta
###Detection des pics delta de la même manière que l'on a détecté les sharp waves ripples
VO=vect_detect_pic(charO,charA,T,'delta',2,100)
#print(VO)
VB=vect_detect_pic(charB,charA,T,'ripples',3,50)
#fich4=open("Vect_pic"+charO[66:-4]+".txt","w")
#fich4.write('Detection de pic en amplitude \n')
#fich4.write(charO[66:-4]+'\n')
#for elem in VO : 
#    #fich4.write(str(elem)+'\n')
#fich4.close()
#Construction d'un vecteur de 0 et de 1 caractérisant la présence de pic delta


#vecteur pour la phase : 
#phi=phase_delta(charB,charA,charO,T,3,15) 
#fich5=open("Phase_"+charO[66:-4]+".txt","w")
#fich5.write('Phase \n')
#fich5.write(charO[66:-4]+'\n')
#for elem in phi : 
#    #fich5.write(str(elem)+'\n')
#fich5.close()

#On passe maintenant à la phase de comparaison de ces séries temporelles avec un fast dynamic Time Warping disponiable sur Python

#Il faut que ce soit des tableaux à 2D : pour un point i (x,y) y valeur de la liste qu'on analyse et x le tems

#Puissance brut
# vect_PB=np.zeros((len(PO),2))#size ligne
# vect_PO=np.zeros((len(PO),2))
# print("PO",len(PO))
# for i in range(len(PO)): #il s'agit de la plus petite liste, on peut simplifier avec des zip surement
#     #print(i)
#     vect_PB[i,0]=T[i]
#     vect_PO[i,0]=T[i]
#     vect_PB[i,1]=PB[i]*100 #permet de comparer sur des amplitudes d'un même ordre de grandeur
#     vect_PO[i,1]=PO[i]
# 
# 
# 
# distance,path=fastdtw(vect_PB, vect_PO,radius=1, dist=euclidean) #distance en quoi ? 
# #print(distance)
# index_a,index_b=zip(*path)
# plt.plot(vect_PB[:,0],vect_PB[:,1],color='blue')
# plt.plot(vect_PO[:,0],vect_PO[:,1],color='red')
# for i in index_a:
#     if (i%100)==0:
#         x1=vect_PB[i,0]
#         y1=vect_PB[i,1]
#         x2=vect_PO[i,0]
#         y2=vect_PO[i,1]
#         plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
# plt.show()

#Pour les vecteurs de 0 et de 1 signalant présence de pic

vect_picO=np.zeros((size,2))
vect_picB=np.zeros((size,2))
#print(len(VO),len(VB),size)
for i in range(size): #il s'agit de la plus petite liste, on peut simplifier avec des zip surement
    #print(i)
    vect_picB[i,0]=T[i]
    vect_picO[i,0]=T[i]
    vect_picB[i,1]=VB[i]
    vect_picO[i,1]=VO[i]

distance,path=fastdtw(vect_picB ,vect_picO,radius=1, dist=euclidean) #distance en quoi ? 
#print(distance)
index_a,index_b=zip(*path)
plt.plot(vect_picB[:,0],vect_picB[:,1],color='blue')
plt.plot(vect_picO[:,0],vect_picO[:,1],color='red')
for i in index_a:
    if (i%100)==0:
        x1=vect_picB[i,0]
        y1=vect_picB[i,1]
        x2=vect_picO[i,0]
        y2=vect_picO[i,1]
        plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
plt.show()