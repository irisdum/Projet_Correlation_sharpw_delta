#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:28:31 2018

@author: iris
"""

from Traitement_fich import*
import matplotlib.pyplot as plt
time=120
chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/' #à changer selon les ordinateurs

#T=open_data(chemin+'BP1-BP2_Temps.txt')[0:time*512] 
T=[round(i/512,6) for i in range(1,time*512+1)]
char_A=chemin+"A'2-A'1_300s.txt" #a changer en fonction de la taille de l'extrait considéré
char_B =chemin+"B'2-B'1_300s.txt"
char_O=chemin+"O'9-O'8_300s.txt"
##Affichons les signaux bruts
# plt.figure()
# plt.plot(T,open_data(chemin+"O'9-O'8_120s.txt")[:512*time])
# plt.show()

##Affichons les signaux filtrés avec signal brut

plt.figure(figsize=(15,30))
# plt.subplot(2,1,1)
# plt.plot(T,open_data(char_O)[:512*time])
# plt.title('signal brut')
# 
# filtre(char_O,T,opt='delta')
# 
# plt.show()

## Affichons puissance et filtre
plt.figure(figsize=(15,30))
plt.subplot(2,1,1)
plot_filtre(char_B,T,opt='ripples')
#plt.subplot(2,1,2)
aff_puiss(char_B,T,1,'ripples')
plt.show()

##Affichons les corrélations entre deux signaux : 
#nom1="B3-B2N3.txt"
#nom2="B4-B3N3.txt"
#Y1=open_data(chemin+"B3-B2N3.txt")[0:20*512]
#Y2=open_data(chemin+"B4-B3N3.txt")[0:20*512]
#print(compare(Y1,Y2,nom1[:-4]+nom2[:-4],'brut'))

##Affichons les signaux filtrés et la puissance calculée

#aff_puiss(chemin+"O'9-O'8_120s.txt",T,h=20,opt='delta')

##Observation de pics delta
#detec_pic(char_O,char_A,T,'delta',3,20)

##Compare les signaux filtrées venant de l'hippocampe et electrode O9-O8:
#compar_delta_ripples(T,chemin+"B3-B2N3.txt",chemin+"O'9-O'8N3.txt")

#Correlation en puissance entre electrode de l'hippocampe
# 
# corr_sig_puiss(chemin+"B'2-B'1N3.txt",chemin+"B'4-B'3N3.txt",T)  #penser à modifier un bout du code. enlever l'option delta

##Correlation en puissance : Calcule la correlation entre deux signaux. Le premier contient des sharpwaves l'autre pour les delta
#filtre
#corr_sig_puiss(char_B,chemin+"O'9-O'8_120s.txt",char_A,T) 


## Eventuelle périodicité entre les signaux bruts O'9-O'8 et B'2-B'1
##Calcule de cette période : on se place soit dans le cas de décalage positif ou négatif 
#print(np.mean(periode_corr(chemin+"B'2-B'1N3.txt",chemin+"O'9-O'8N3.txt",0.10,1,0)))

##Etudions si les sharpw sont détécté à peu près au même moment pour deux electrodes proches dans l'hippocampe
#detec_pic(char_B,char_A,T,'ripples',3,9,1)
#detec_pic(chemin+"B'2-B'1N3.txt",char_A,T,'ripples',3,50,1)

#print(detec_pic(chemin+"B3-B2N3_120s.txt",char_A,T)[0])
#Etudions les differents type de sharpw soit trié en fonction de leur écart à la moyenne
#sort_sharpw_ripples(chemin+"B'2-B'1N3_120s.txt",char_A,'ripples',1)
#sort_sharpw_ripples(char_B,char_A,T,'ripples',1)
#aff_puiss(char_A,T,20,'ripples')
#aff_puiss(chemin+"O'9-O'8_120s.txt",T,20,'delta')
##Etudions si des sharps waves rippples coincident graphiquement avec l'apparition de delta

#detec_delta_sharp_ripples(chemin+"O'9-O'8N3.txt",char_A,chemin+"B'2-B'1N3.txt")

##Verifions que les pics observé correspondent bien à des sharpw et non des pics epileptique

#detec_epileptic_pic(char_A,char_B,T)
#clean_epileptic_pic(chemin+ "A'2-A'1N3.txt",chemin+"B'4-B'3N3.txt",T,1)
#récupérons la  du signal lorsqu'un sharpw est détecté
#phase_delta(chemin+"B'3-B'2N3.txt",char_A,chemin+"O'9-O'8N3.txt",T,3,5)
#Determinons les valeurs des phases trouvées en fonction de l'amplitude du sharpw détectées
#stat_phase(chemin+"B'2-B'1N3_60s.txt",char_A,chemin+"O'9-O'8N3_60s.txt",T)

#Comparons les vecteurs detectiion pour delta et sharpw
# charB=chemin+"B'2-B'1N3_120s.txt"
# # charO=chemin+"O'9-O'8_120s.txt"
# # plt.figure(figsize=(30,15))
# plt.figure()
# plt.subplot(2,1,1)
#VB=vect_detect_pic(char_B,char_A,T,'ripples',3,15,1)
# plt.plot(T[:(len(VB))],VB)
# plt.subplot(2,1,2)
# plt.plot(T,vect_detect_pic_STA(char_B,char_A,T,opt='ripples',fact=3,max_fact=15,h=1))
# plt.show()
# plt.show()
# # VO=vect_detect_pic(charO,T,'delta',2,100,1)
# plt.legend(loc=3)



#Etablissons les statistiques : On essaie de se focaliser sur des extraits de 1 à 2 minutes
# char_B=chemin+"B3-B2N3_120s.txt"
# char_Bp=chemin+"B'3-B'2N3_120s.txt"
# print("Cas de "+ char_B[66:-4])
# statistic_sharpw(char_B,char_A,T,3,10,20)
# print("Cas de "+ char_Bp[66:-4])
# statistic_sharpw(char_Bp,char_A,T,3,10,20)

##Verifions la convolution : 
# X=vect_detect_pic_STA(char_B,char_A,T,opt='ripples',fact=3,max_fact=10,h=1)
# convolve_spike(T,X)
#Fonctionne mais il y a un petit décalage : voir comment faire... 

##Comparaison pic delta et présence SPW-Rs
# plt.figure()
# plt.title("Présence rythmes delta et SPW-Rs")
# VB=vect_detect_pic(char_B,char_A,T,'ripples',5,50,1)
# VO=vect_detect_pic(char_O,char_A,T,'delta',2,50,1)
# plt.plot([i/512 for i in range(len(VO))],VO,label='rythmes delta')
# plt.plot([i/512 for i in range(len(VB))],VB,label='SPW-Rs')
# plt.xlabel("Temps en seconde")
# plt.legend(loc=3)
# plt.show()  