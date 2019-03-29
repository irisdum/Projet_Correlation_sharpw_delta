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
from STA import*
import ast
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth # for kdtree method
from scipy.cluster.hierarchy import*
from scipy.spatial.distance import pdist
#import panda #open source library for data analysis

#Define variable
time=300 #in second
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
            print(i)
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
    Ncluster=4

    print(len(D_high),len(A_high),len(A_low))
    X=np.array([D_high,A_high,A_low])#quantitatives variable
    X=X.T # what does this line ???
    print('X',type(X),np.size(X))
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
    col=['r.','b.','g.','r.']
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
    
def extract_gr(name_sig,T,num_gr):
    """returns a vector with the index of the spike we keep"""
    predictions,D_high,A_high,A_low=cluster(name_sig,T)
    l_spike=[]
    for i in range(len(prediction)):
        if prediction==num_gr:
            l_spike+=[t_max_pic_high[i]]
    return l_spike
    
def asc_cluster(name_sig,T,method='ward',disp=True,nb_br=4):
    """ Applique une méthode ascendante nos données"""
    nb_gr=4 #we choose two groups
    D_high,sig_high,sig_low,t_max_pic_high,ind,A_high,A_low=crit_event(name_sig,T)
    #print(t_max_pic_high)
    X=np.array([D_high,A_high,A_low])#quantitatives variable
    X=X.T
    #print(np.size(X))
    Z = linkage(X, method)
    #Have to be close to 1 : 
    c, coph_dists = cophenet(Z, pdist(X))
    print('cophenecy',c)
    cluster=fcluster(Z,nb_gr,criterion='maxclust')
    dist=maxdists(Z)
    if disp:
        
        plt.scatter(X[:,1],X[:,2])
        plt.show()
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()
        plt.figure(figsize=(10, 8))
        inconsistence=inconsistent(Z)
        bool=is_valid_linkage(Z)
    
        plt.scatter(X[:,0], X[:,1], c=cluster)  # plot points with cluster dependent colors
        plt.title('A-highen fonction de D_high')
        plt.show()
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:,1], X[:,2], c=cluster)  # plot points with cluster dependent colors
        plt.title('A_low en fonction de A_high')
        plt.show()
    
    return cluster,t_max_pic_high,ind
    
def sort_sharpw_cluster(name_sig,T,gr,nb_gr=4):
    cluster,t_max_pic_high,ind=asc_cluster(name_sig,T,'complete',False,nb_gr)
    t_max_gr=[]
    for i in range(len(t_max_pic_high)) : 
        if cluster[i]==gr:
            t_max_gr+=[t_max_pic_high[i]]
    return t_max_gr
    
    
def vect_detect_pic_STA2(name_sig,T,gr=1,graph=True) : 
    """retourne un vecteur de 0 et de 1, mais 1 pour le tmax du sharpw  0 sinon et non pas des paliers comme dans la fonction vect_detec_pic"""
        
    U0=sort_sharpw_cluster(name_sig,T,gr)
    vect=[]#liste qui ca contenir les 0 et les 1
    i=0
    #print("taille T", len(T))   
    if U0==[]:
        #print('pas de pic')
        return [0 for i in range(len(T))]   
    for elem in T:
        compt=0 #détecte si on a trouvé un pic
        if i<len(U0):# qd on a pas encore détecté tous les sharpw ripples
            for i in range(len(U0)):
                if round(elem,5)==round(U0[i],5):#on a le max d'un pic
                    vect+=[1]
                    i+=1
                    compt+=1
                    #print(i)
            if compt==0:
                vect+=[0]
        else:
            vect+=[0]
    if graph:
        plt.figure()
        plt.plot(T,vect)
        plt.show()
    return vect

def plot_all_pic_cluster(name_sig,T,nb_cluster=4):
    lcol=['blue','red','green','black']
    plt.figure(figsize=(15,30))
    for i in range(nb_cluster):
        plt.plot(T,vect_detect_pic_STA(name_sig,T,gr=i,graph=False),lcol[i])
    plt.show()
    return False
#cluster(char_B,T)    
#plot_influ_crit(char_B,T)
#plot_ampl(char_B,T)
#plot_duree(char_B,T)

#asc_cluster(char_B,T,'complete',True) #ward average, complete
#print(sort_sharpw_cluster(char_B,T,1))
#vect_detect_pic_STA(char_B,T,1,True)
# plot_all_pic_cluster(char_B,T,nb_cluster=4)

##Testons STA
# 
# # 
# # Matrice X qui contient phase des rythmes delta ainsi que leur puissance dans le delta temps  précédant l'apparition du pic
# #puiss=normalise_puiss(charO,T,1,'delta')
# puiss=calc_puiss(char_O,T,1,'delta')[0]
# size=len(puiss)
# liste_t=sort_sharpw_cluster(char_B,T,1,2)
# #vect_1=vect_detect_pic(charO,charA,T,opt='ripples',fact=2,max_fact=10,h=1)
# #size=len(vect_1)
# # print("size",size)
# X=np.zeros((2,size))
# X[0,:]=vect_phase(charO,T)[:size] #Car la taille de la fenêtre de la puissance fait en sorte qu'on ne peut calculer la puissance sur tous les derniers points du vecteurs
# X[1,:]=puiss
# delta = 512
# #For the phase signal
# Liste_val_mean,Liste_std=STA_dt_after(liste_t,X,delta)
# print(np.size(Liste_val_mean,2))
# Liste_val_mean=Liste_val_mean[0,:]
# Liste_val_std=Liste_val_mean[0,:]
# list_val_std=[i+j for i,j in zip(Liste_std,Liste_val_mean)] #std from the new values calculated
# list_val_std2=[j-i for i,j in zip(Liste_std,Liste_val_mean)]
# 
# #For the power signal 
# Liste_val_mean_pow,Liste_std_pow=STA_dt_before(liste_t,X,delta)
# Liste_std_pow=Liste_std_pow[1,:]
# Liste_val_mean_pow=Liste_val_mean_pow[1,:]
# list_val_std_pow=[i+j for i,j in zip(Liste_std_pow,Liste_val_mean_pow)] #std from the new values calculated
# list_val_std2_pow=[j-i for i,j in zip(Liste_std_pow,Liste_val_mean_pow)]
# # 
# # 
# fig=plt.figure(figsize=(15,30))
# fig.suptitle("Motif moyen avant sharpw pour "+charB[66:-4],fontsize=16)
# # # 
# # # plt.grid()
# # # plt.plot([i/512 for i in range(delta)],Liste_val_mean,label="Vect_detect_pic")
# # # plt.plot([i/512 for i in range(delta)],list_val_std_pow,c='r',label="std")
# # # plt.plot([i/512 for i in range(delta)],list_val_std2_pow,c='r',label="std")
# plt.subplot(2,1,1)
# 
# plt.title("Phase ryhtmes delta")
# plt.plot([i/512 for i in range(-delta,0)],Liste_val_mean,label="average phase")
# #Print std for each values 
# plt.plot([i/512 for i in range(-delta,0)],list_val_std,c='r',label="std")
# plt.plot([i/512 for i in range(-delta,0)],list_val_std2,c='r',label="std")
# plt.legend(loc=3)
# plt.grid()
# plt.subplot(2,1,2)
# plt.title("Puissance")
# plt.plot([i/512 for i in range(-delta,0)],Liste_val_mean_pow,label="average power")
# plt.grid()
# plt.plot([i/512 for i in range(-delta,0)],list_val_std_pow,c='r',label="std")
# plt.plot([i/512 for i in range(-delta,0)],list_val_std2_pow,c='r',label="std")
# plt.xlabel("Temps en seconde")
# 
# plt.legend(loc=3)
# plt.show() 

##Pour le vecteur de 0 et de 1 : 
vect_1=vect_detect_pic_STA(char_O,T,opt='ripples',fact=3,max_fact=10,h=1) # regarde comportement rythme delta
print('len vec1',len(vect_1))
#print(len(vect_1),'nb pic delta')
size=len(vect_1)
# print("size",size)
X1=np.zeros((1,size))

X1[0,:]=vect_1
#X1[:,1]=vect_1
delta=512
liste_t=sort_sharpw_cluster(char_B,T,2,2)
#liste_t contains tmax of sharpw detection
p_SW=len(liste_t)/(time*512) #proba d'observer un rythme SW
print('n_delta',sum(vect_1)/(time*512))
Liste_val_mean,Liste_std=STA_dt_after(liste_t,X1,delta)
#print('len val mean', np.size(Liste_std))
#Liste_val_mean=[i for i in Liste_val_mean1]
# #print("taille ecart-type" ,Liste_std)
list_val_std=[i+j for i,j in zip(Liste_std,Liste_val_mean)] #std from the new values calculated
list_val_std2=[j-i for i,j in zip(Liste_std,Liste_val_mean)]
Liste_val_mean2=STA_dt_before(liste_t,X1,delta)[0]


Liste_std2=STA_dt_before(liste_t,X1,delta)[1]
# #print("taille ecart-type" ,Liste_std)
list_val_std3=[i+j for i,j in zip(Liste_std2,Liste_val_mean2)] #std from the new values calculated
list_val_std4=[j-i for i,j in zip(Liste_std2,Liste_val_mean2)]
plt.figure(figsize=(15,30))
plt.subplot(2,1,1)
#print('plot pb', np.size(np.transpose(list_val_std),1),len([i/512 for i in range(delta)]))

#plt.plot([i/512 for i in range(delta)],np.transpose(list_val_std),c='r')
plt.plot([i/512 for i in range(delta)],np.transpose(Liste_val_mean),c='b')
#plt.plot([i/512 for i in range(delta)],np.transpose(list_val_std2),c='r',label="std")
plt.title("STA"+charB[66:-4]+'groupe 1/2' )
plt.plot([i/512 for i in range(delta)],[sum(vect_1)/(time*512) for i in range(delta)])
plt.subplot(2,1,2)
#plt.plot([-i/512 for i in reversed(range(delta))],np.transpose(list_val_std3),c='r')
plt.plot([-i/512 for i in reversed(range(delta))],np.transpose(Liste_val_mean2),c='b',label="mean")

#plt.plot([-i/512 for i in reversed(range(delta))],np.transpose(list_val_std4),c='r',label="std")
plt.xlabel("en secondes")
plt.show()