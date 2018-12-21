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
charB=chemin+"B2-B1_300s.txt"
charO=chemin+"O'9-O'8_300s.txt"
charA=chemin+"A'2-A'1_300s.txt"
time=300
# Definissions un vecteur Y de 0 et de 1 contenant les moments d'apparition des pics : 
T=[round(i/512,6) for i in range(1,time*512+1)]


# Matrice X qui contient phase des rythmes delta ainsi que leur puissance dans le delta temps  précédant l'apparition du pic
puiss=normalise_puiss(charO,T,1,'delta')
size=len(puiss)
print("size",size)
X=np.zeros((2,size))
X[0,:]=vect_phase(charO,T)[:size] #Car la taille de la fenêtre de la puissance fait en sorte qu'on ne peut calculer la puissance sur tous les derniers points du vecteurs
X[1,:]=puiss

#Y=np.array(vect_detect_pic_STA(charB,charA,T,'ripples',3,50,1)[:size]) #On limite la taille car fenetre de calcul de la puissance plus petite pour B que pour O
#Implementons le spike trigerred average : d'après wikipedia

def STA(Y,X):
    """Spike trigerred average 
    Y = vecteur de 0 et de 1, il s'agit de comparer les valeurs instantanée
    """
    global size
    n=sum(Y) #nber of spike
    sizeX=X.shape[0]
    print("nombre de pic vu",n)
    print("Etape 1 ")
    Mat=np.array((sizeX,sizeX))
    Mat=(X.dot(X.transpose())) #Taille de matrice trop importante voir comment réduire le temps de calcul
    print("Etape 2")
    #print(Mat.shape)
    Mat_inv=scipy.linalg.inv(Mat)
    print("Etape 3")
    Mat2=X.dot(Y.transpose())
    print("Etape 4")
    Avg=size/n*Mat_inv.dot(Mat2)
    return Avg

#print("Pas satisfait du réslutat pour le moment", STA(Y,X))
    
##Calculons STA sur une fenêtre de  256 points

delta =256 #points soit 500 ms
liste_t=detec_pic(charB,charA,T,opt='ripples',fact=3,max_fact=10,h=1)[0] #contains tmax of sharpw detection

def STA_dt(t_sharpw,X,dt=256):
    """Spike triggered average
    Gives the value of the average xi for a window of 500ms before the detection of sharw
    t_sharpw : list of time containing the sharpw
    X: Array of size (1,len)
    delta : size of the window"""
    n=len(t_sharpw) #nber of spike
    print("nb de pic", n)
    size_x=X.shape[0]
    list_sub_matX=[]
    M_mean=np.zeros((size_x,dt)) #contains the value of the average xj for param i
    M_std=np.zeros((size_x,dt))#contains the value of the std of xj
    for k in range(len(liste_t)):
        ind_begin=int(liste_t[k]*512-dt) #index to begin the extract of the matrice
        if ind_begin>0 and liste_t[k]*512<X.shape[1]:#we separate
            #print("debut fin max ",ind_begin,int(liste_t[k]*512),X.shape[1])
            Sub_MatX=X[:,ind_begin:int(liste_t[k]*512)]
            #print(Sub_MatX.shape)
            #M_mean=M_mean+Sub_MatX
            #print(k)
            list_sub_matX+=[Sub_MatX]
    for i in range(size_x):#for all the parameters
        for j in range(dt):#all the xj 
            value=[]
            for elem in list_sub_matX: #goes over all the submatrice
                value+=[elem[i,j]]
            M_mean[i,j]=np.mean(value)
        #print("size M_Mean",)
            M_std[i,j]=np.std(value)
    return M_mean, M_std 
                
#print(STA_dt(liste_t,X,256)) 
Liste_val_mean=STA_dt(liste_t,X,256)[0][1,:]
Liste_std=STA_dt(liste_t,X,256)[1][1,:]
#print("taille ecart-type" ,Liste_std)
list_val_std=[i+j for i,j in zip(Liste_std,Liste_val_mean)]
list_val_std2=[j-i for i,j in zip(Liste_std,Liste_val_mean)]

plt.figure()
plt.plot([i/512 for i in range(256)],Liste_val_mean)
#Print std for each values 
plt.plot([i/512 for i in range(256)],list_val_std,c='r')
plt.plot([i/512 for i in range(256)],list_val_std2,c='r')
plt.show()       
        