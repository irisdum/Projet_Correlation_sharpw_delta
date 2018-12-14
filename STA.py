#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''@author: iris'''

# Spike trigerred Average

from Traitement_fich import*
import numpy as np

import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy
chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/'
charB=chemin+"B'2-B'1N3_120s.txt"
charO=chemin+"O'9-O'8_120s.txt"
charA=chemin+"A'2-A'1_120s.txt"
time=60
# Definissions un vecteur Y de 0 et de 1 contenant les moments d'apparition des pics : 
T=[round(i/512,6) for i in range(1,time*512+1)]


# Matrice X qui contient phase des rythmes delta ainsi que leur puissance dans le delta temps  précédant l'apparition du pic
puiss=calc_puiss(charO,T,1,'delta')[0]
size=len(puiss)
X=np.zeros((2,size))
X[0,:]=vect_phase(charO,T)[:size] #Car la taille de la fenêtre de la puissance fait en sorte qu'on ne peut calculer la puissance sur tous les derniers points du vecteurs
X[1,:]=puiss

Y=vect_detect_pic_STA(charB,charA,T,'ripples',3,50,1)[:size] #On limite la taille car fenetre de calcul de la puissance plus petite pour B que pour O
#Implementons le spike trigerred average : 
delta=102 #points soit un peu moins de 200 ms (freq de 512)
#Pour chaque ligne de X on calcule la moyenne des grandeus sur une fenêtre de 102 points avant le sharpw

def STA(Y,X,dt=102):
    """Spike trigerred average 
    Y = vecteur de 0 et de 1
    dt taille de la fenêtre en nombre de point"""
    global size
    n=sum(Y) #nber of spike
    sizeX=X.shape[0]
    # Avg_response=np.zeros((X.shape[0],sum(Y))
    # for y in range(len(Y)):
    #     if Y[y]==1:#on calcule la moyenne des valeurs
    #             if y>dt:# evite out of range
    #                 for i in range(y-dt,y):
    #                     Avg_response[0,y]+=Avg_response[0,y]
    #                     Avg_response[1,y]+=Avg_response[1,y]
    #                 Avg_response[1,y]=Avg_response[1,y]/dt #on moyenne
    #                 Avg_response[0,y]=Avg_response[1,y]/dt #on moyenne
    Mat=np.array((sizeX,sizeX))
    Mat=(X.transpose().dot(X))
    Mat_inv=np.linalg(Mat)
    Mat2=X.tranpose().dot(Y)
    Avg=Mat_inv.dot(Mat2)
    
    return Avg
    
print(STA(Y,X,delta))
    

