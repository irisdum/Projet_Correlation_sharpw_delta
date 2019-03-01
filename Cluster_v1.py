#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Feb 26 14:52:06 2019

@author: iris dumeur
"""
# Cluster : Amplitude and time v1

#Import 
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from Traitement_fich import*
import ast
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth # for kdtree method

#Define variable
time=50 #in second
chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/' #change depending the data set

T=[round(i/512,6) for i in range(1,time*512+1)] #contains time 
char_A=chemin+"A'2-A'1_300s.txt" #a changer en fonction de la taille de l'extrait considéré
char_B =chemin+"B'2-B'1_300s.txt"
char_O=chemin+"O'9-O'8_300s.txt"



def duree_event(name_sig,T):
    """duration of the SPW for high freq : 120-250 Hz
    low  : 10-80 Hz """
    sig_high=filtre(name_sig,T,'ripples')
    sig_low=filtre(name_sig,T,'epileptic')
    t_max_pic_high,p_max_high,inter_pic_high,int_high,ind=detec_pic(name_sig,T,'ripples',1,70,1) #Critère detection bas
    #t_max_pic_low,p_max_low,inter_pic_low,int_low=detec_pic(name_sig,T,'epileptic',1,50,1) #Critère detection bas
    
    D_high=[len(elem)/512 for elem in int_high]
    return D_high,sig_high,sig_low,t_max_pic_high,ind

def plot_duree(name_sig,T):
    D_high,sig_high,sig_low,t_max_pic_high,ind=duree_event(name_sig,T)
    fig, axs = plt.subplots(3, 1)
    fig.canvas.set_window_title("ADuree"+name_sig[66:-4])
    fig.suptitle('Duree high', fontsize=16)
    puiss,Tpuiss=calc_puiss(name_sig,T,1)
    axs[0].plot(Tpuiss,puiss)
    axs[0].plot(t_max_pic_high,[0]*len(t_max_pic_high),'r*')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_title("Puissance")
    axs[1].plot(T,sig_high)
    axs[1].set_title("120-250 Hz")
    axs[2].set_title('Duree high')
    axs[2].plot(t_max_pic_high,D_high,'b')
    plt.show()
def crit_event(name_sig,T):
    """A partir du signal filtré et de la liste avec indice debut et fin de chaque evenement on retourne amplitude de l'evenement"""
    D_high,sig_high,sig_low,t_max_pic_high,ind=duree_event(name_sig,T)
    A_high=[]
    A_low=[]
    #print(ind[2][0])
    i=0
    taille_ind=len(ind)
    while i < taille_ind : 
        # print(ind[i])
        if ind[i][0] != ind[i][1] : 
            l_high=sig_high[ind[i][0]:ind[i][1]]
            l_low=sig_low[ind[i][0]:ind[i][1]]
            A_high+=[max(l_high)-min(l_high)] 
           # A_high+=[np.linalg.norm(l_high)]
            A_low+=[max(l_low)-min(l_low)]
        else:
            print('étroit')
            # del ind[i]
            # del D_high[i]
            # del t_max_pic_high[i]
            taille_ind-=1
        i+=1
    return D_high,sig_high,sig_low,t_max_pic_high,ind,A_high,A_low
    


def plot_ampl(name_sig,T):
    """Affiche l'amplitude caculé pour chaque pic"""
    D_high,sig_high,sig_low,t_max_pic_high,ind,A_high,A_low=crit_event(name_sig,T)
    A_high2=[0]*len(T)
    for i in range(len(T)):
        if i==ind[0]:
            print(i)
            del ind[0]
            A_high2[i]=A_high[0]
            del A_high[0]
    fig, axs = plt.subplots(3, 1)
    fig.canvas.set_window_title("Amplitude"+name_sig[66:-4])
    fig.suptitle('Amplitude', fontsize=16)
    puiss,Tpuiss=calc_puiss(name_sig,T,1)
    axs[0].plot(Tpuiss,puiss)
    axs[0].plot(t_max_pic_high,[0]*len(t_max_pic_high),'r*')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_title("Puissance")
    axs[1].plot(T,sig_high)
    axs[1].set_title("120-250 Hz")
    axs[2].set_title('Amplitude high')
    axs[2].plot(t_max_pic_high,A_high,'b')
    
    plt.show()
            
# Clustering function 

def cluster(name_sig,T):

    D_high,sig_high,sig_low,t_max_pic_high,ind,A_high,A_low=crit_event(name_sig,T)
    Ncluster=2

   
    X=np.array([D_high,A_high,A_low])#quantitatives variable
    X=X.T # what does this line ???
    kmeans = KMeans(n_clusters=Ncluster,n_init=200).fit(X)
    print(kmeans.cluster_centers_)
    predictions=kmeans.predict(X)
    fig, axs = plt.subplots(3, 1)
    fig.canvas.set_window_title("Cluster"+name_sig[66:-4])
    fig.suptitle('Clustering', fontsize=16)
    
    axs[0].plot(T,sig_high)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_title("20-250 Hz")
   
    axs[1].plot(T,sig_low)
    axs[1].set_title("10-80 Hz")
    axs[2].set_title('Attributions des groupes')
    axs[0].plot(t_max_pic_high,predictions,'r.')
    axs[1].plot(t_max_pic_high,predictions,'r.')
    axs[2].plot(t_max_pic_high,predictions,'r.')
    
    plt.show()
    
  
    return predictions,D_high,A_high,A_low
    
    
  
def plot_influ_crit(name_sig,T):
    pred,D_high,A_high,A_low=cluster(name_sig,T)
    plt.figure()
    col=['r.','b.','g.']
    fig, axs = plt.subplots(1, 3)
    fig.canvas.set_window_title("Cluster_influence_crit"+name_sig[66:-4])
    fig.suptitle('Influence criteres classification', fontsize=16)
    #print(len(D),len(pred))
    
    for i in range(len(pred)):
        axs[0].plot(pred[i],D_high[i],col[pred[i]])
        axs[1].plot(pred[i],A_high[i],col[pred[i]])
        axs[2].plot(pred[i],A_low[i],col[pred[i]])
   
    axs[0].set_xlabel("Groupe")
    axs[0].set_title("Duree 120-250 Hz")
    axs[1].set_xlabel("Groupe")
    axs[1].set_title("Amplitude 120-250 Hz")
    axs[2].set_xlabel("Groupe")
    axs[2].set_title("Amplitude 10-80 Hz")
    #plt.ylabel("Duree de l'evenement")
    plt.show()
    
#cluster(char_B,T)    
plot_influ_crit(char_B,T)
#plot_ampl(char_B,T)
#plot_duree(char_B,T)