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
charB=chemin+"B'2-B'1_300s.txt"
charO=chemin+"O'9-O'8_300s.txt"
charA=chemin+"A'2-A'1_300s.txt"
time=50
# Definissions un vecteur Y de 0 et de 1 contenant les moments d'apparition des pics : 
T=[round(i/512,6) for i in range(1,time*512+1)]


# Matrice X qui contient phase des rythmes delta ainsi que leur puissance dans le delta temps  précédant l'apparition du pic
#puiss=normalise_puiss(charO,T,1,'delta')
puiss=calc_puiss(charO,T,1,'delta')[0]
size=len(puiss)
#vect_1=vect_detect_pic(charO,charA,T,opt='ripples',fact=2,max_fact=10,h=1)
#size=len(vect_1)
# print("size",size)
X=np.zeros((2,size))
X[0,:]=vect_phase(charO,T)[:size] #Car la taille de la fenêtre de la puissance fait en sorte qu'on ne peut calculer la puissance sur tous les derniers points du vecteurs
X[1,:]=puiss

# X[0,:]=vect_1
# X[1,:]=vect_1
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

#delta =256 #points soit 500 ms
liste_t=detec_pic(charB,charA,T,opt='ripples',fact=5,max_fact=10,h=1)[0] #contains tmax of sharpw detection

def STA_dt_before(t_sharpw,X,dt=128):
    """Spike triggered average
    Gives the value of the average xi for a window of 500ms before the detection of sharw
    t_sharpw : list of time containing the sharpw
    X: Array containing the signals to study one row= one signal
    delta : size of the window we study the beahviour of the siganls X before a sharpw is detected"""
    n=len(t_sharpw) #nber of spike
    print("nb de pic", n)
    size_x=X.shape[0] #comment when X size 1
    #size_y=X.shape[1]
    #size_x=1
    #size_y=len(X)
    
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

            M_std[i,j]=np.std(value)
    return M_mean, M_std 
            
def STA_dt_after(t_sharpw,X,dt=128):
    """Spike triggered average
    Gives the value of the average xi for a window of 500ms before the detection of sharw
    t_sharpw : list of time containing the sharpw
    X: Array containing the signals to study one row= one signal
    delta : size of the window we study the beahviour of the siganls X after a sharpw is detected"""
    n=len(t_sharpw) #nber of spike
    print("nb de pic", n)
    size_x=X.shape[0]
    list_sub_matX=[]
    M_mean=np.zeros((size_x,dt)) #contains the value of the average xj for param i
    M_std=np.zeros((size_x,dt))#contains the value of the std of xj
    for k in range(len(liste_t)):
        ind_begin=int(liste_t[k]*512) #index to begin the extract of the matrice
        if ind_begin>0 and liste_t[k]*512+dt<X.shape[1]:
            #print("debut fin max ",ind_begin,int(liste_t[k]*512),X.shape[1])
            Sub_MatX=X[:,ind_begin:int(liste_t[k]*512 +dt)]
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
            M_std[i,j]=np.std(value)
    return M_mean, M_std 
    
##STA Before sharpw results
#print(STA_dt(liste_t,X,256)) 
#For the phase signal
delta=512
Liste_val_mean=STA_dt_before(liste_t,X,delta)[0][0,:]
Liste_std=STA_dt_before(liste_t,X,delta)[1][0,:]
#print("taille ecart-type" ,Liste_std)
list_val_std=[i+j for i,j in zip(Liste_std,Liste_val_mean)] #std from the new values calculated
list_val_std2=[j-i for i,j in zip(Liste_std,Liste_val_mean)]

#For the power signal 
Liste_val_mean_pow=STA_dt_before(liste_t,X,delta)[0][1,:]
Liste_std_pow=STA_dt_before(liste_t,X,delta)[1][1,:]

list_val_std_pow=[i+j for i,j in zip(Liste_std_pow,Liste_val_mean_pow)] #std from the new values calculated
list_val_std2_pow=[j-i for i,j in zip(Liste_std_pow,Liste_val_mean_pow)]
# 
# 
fig=plt.figure(figsize=(15,30))
fig.suptitle("Motif moyen avant sharpw pour "+charB[66:-4],fontsize=16)
# # 
# # plt.grid()
# # plt.plot([i/512 for i in range(delta)],Liste_val_mean,label="Vect_detect_pic")
# # plt.plot([i/512 for i in range(delta)],list_val_std_pow,c='r',label="std")
# # plt.plot([i/512 for i in range(delta)],list_val_std2_pow,c='r',label="std")
plt.subplot(2,1,1)

plt.title("Phase ryhtmes delta")
plt.plot([i/512 for i in range(-delta,0)],Liste_val_mean,label="average phase")
#Print std for each values 
plt.plot([i/512 for i in range(-delta,0)],list_val_std,c='r',label="std")
plt.plot([i/512 for i in range(-delta,0)],list_val_std2,c='r',label="std")
plt.legend(loc=3)
plt.grid()
plt.subplot(2,1,2)
plt.title("Puissance")
plt.plot([i/512 for i in range(-delta,0)],Liste_val_mean_pow,label="average power")
plt.grid()
plt.plot([i/512 for i in range(-delta,0)],list_val_std_pow,c='r',label="std")
plt.plot([i/512 for i in range(-delta,0)],list_val_std2_pow,c='r',label="std")
plt.xlabel("Temps en seconde")

plt.legend(loc=3)
plt.show()       
# #     
##STA after sharpw results
Liste_val_mean=STA_dt_after(liste_t,X,delta)[0][0,:]
Liste_std=STA_dt_after(liste_t,X,delta)[1][0,:]
# #print("taille ecart-type" ,Liste_std)
list_val_std=[i+j for i,j in zip(Liste_std,Liste_val_mean)] #std from the new values calculated
#list_val_std2=[j-i for i,j in zip(Liste_std,Liste_val_mean)]

# 
# #For the power signal 
Liste_val_mean_pow=STA_dt_after(liste_t,X,delta)[0][1,:]
Liste_std_pow=STA_dt_after(liste_t,X,delta)[1][1,:]

list_val_std_pow=[i+j for i,j in zip(Liste_std_pow,Liste_val_mean_pow)] #std from the new values calculated
list_val_std2_pow=[j-i for i,j in zip(Liste_std_pow,Liste_val_mean_pow)]
# 
# #Plot 
fig=plt.figure(figsize=(15,30))

fig.suptitle("STA après sharpw pour "+charB[66:-4],fontsize=16)
# plt.grid()
# plt.plot([i/512 for i in range(delta)],Liste_val_mean,label="Vect_detect_pic")
# plt.plot([i/512 for i in range(delta)],list_val_std_pow,c='r',label="std")
# plt.plot([i/512 for i in range(delta)],list_val_std2_pow,c='r',label="std")
# plt.subplot(2,1,1)
plt.subplot(2,1,1)
plt.title("Phase rythmes delta")
plt.plot([i/512 for i in range(delta)],Liste_val_mean,label="average phase")
#Print std for each values 
plt.plot([i/512 for i in range(delta)],list_val_std,c='r',label="std")
plt.plot([i/512 for i in range(delta)],list_val_std2,c='r',label="std")
plt.legend(loc=3)
plt.grid()
plt.subplot(2,1,2)
plt.title("Puissance")
plt.plot([i/512 for i in range(delta)],Liste_val_mean_pow,label="average power")
plt.grid()
plt.plot([i/512 for i in range(delta)],list_val_std_pow,c='r',label="std")
plt.plot([i/512 for i in range(delta)],list_val_std2_pow,c='r',label="std")
plt.xlabel("Temps en seconde")
plt.legend(loc=3)
plt.show()
# 
# plt.figure()
# plt.title("Présence pic delta")
# VB=vect_detect_pic(charB,charA,T,'ripples',5,50,1)
# VO=vect_detect_pic(charO,charA,T,'delta',3,50,1)
# plt.plot([i/512 for i in range(len(VO))],VO,label='rythmes delta')
# plt.plot([i/512 for i in range(len(VB))],VB,label='SPW-Rs')
# plt.xlabel("Temps en seconde")
# plt.legend(loc=3)
# plt.show()  

## Pour vecteur de 0 et de 1 : 
#after

# vect_1=vect_detect_pic(charO,charA,T,opt='ripples',fact=3,max_fact=10,h=1)
# size=len(vect_1)
# # print("size",size)
# X1=np.zeros((1,size))
# 
# X1[0,:]=vect_1
# #X1[:,1]=vect_1
# delta=512
# liste_t=detec_pic(charB,charA,T,opt='ripples',fact=5,max_fact=10,h=1)[0] #contains tmax of sharpw detection
# Liste_val_mean=STA_dt_after(liste_t,X1,delta)[0]
# Liste_std=STA_dt_after(liste_t,X1,delta)[1]
# # #print("taille ecart-type" ,Liste_std)
# list_val_std=[i+j for i,j in zip(Liste_std,Liste_val_mean)] #std from the new values calculated
# list_val_std2=[j-i for i,j in zip(Liste_std,Liste_val_mean)]
# Liste_val_mean2=STA_dt_before(liste_t,X1,delta)[0]
# 
# 
# Liste_std2=STA_dt_before(liste_t,X1,delta)[1]
# # #print("taille ecart-type" ,Liste_std)
# list_val_std3=[i+j for i,j in zip(Liste_std2,Liste_val_mean2)] #std from the new values calculated
# list_val_std4=[j-i for i,j in zip(Liste_std2,Liste_val_mean2)]
# plt.figure(figsize=(15,30))
# plt.subplot(2,1,1)
# plt.plot([i/512 for i in range(delta)],list_val_std,c='r')
# plt.plot([i/512 for i in range(delta)],Liste_val_mean,c='b')
# plt.plot([i/512 for i in range(delta)],list_val_std2,c='r',label="std")
# plt.title("STA"+charB[66:-4] )
# 
# plt.subplot(2,1,2)
# plt.plot([i/512 for i in range(delta)],list_val_std3_,c='r')
# plt.plot([i/512 for i in range(delta)],Liste_val_mean2,c='b',label="mean")
# plt.plot([i/512 for i in range(delta)],list_val_std4,c='r',label="std")
# plt.xlabel("en secondes")
# plt.show()
