#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 4 10:52:06 2019

@author: iris dumeur
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from Traitement_fich import*
import ast
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth # for kdtree method

#Define variable
time=50 #in second
chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/' #à changer selon les ordinateurs

T=[round(i/512,6) for i in range(1,time*512+1)] #contains time 
char_A=chemin+"A'2-A'1_300s.txt" #a changer en fonction de la taille de l'extrait considéré
char_B =chemin+"B'2-B'1_300s.txt"
char_O=chemin+"O'9-O'8_300s.txt"


def descript(sig):
    """ Sig : signal containing sharpw. 
    Descript aim at extracting different values when a spkie occurs"""
    fs=512
    spw_ampl=[] #amplitude spike
    spw_d1=[] #first derivative
    spw_d2=[] #second derivative
    #liste of the sharpw spikes:
    #sig=open_data(name_sig) #return a vector containing the value

    #print(liste_spike)
    #dt_prec=int(20*0.001*fs) #correspond à 4 pts 
    #dt_suiv=int(20*0.001*fs) #correspond à 6 pts
    dt_prec=4
    dt_suiv=6
    dt_energy=int(12*0.001*fs) #correspond à 2 pts
    #print(dt_prec, dt_suiv,dt_energy)
    inter=20 #size of the intervall where we calculate the mean of the interesting variables (E,U,D)
    U1=[]
    U2=[]
    D1=[]
    E1=[]
    D2=[]
    dt=max(dt_prec,dt_suiv,dt_energy)+1+inter #useful for boucle for, avoid index out of range
    print(dt)
    for i in range(dt,len(sig)-dt):
        #ind=peak_ind[i]
        #print(i)
    
        #U1+=[sig[i-dt_prec]-sig[i-dt_prec-1]] #dérivée première avant le point #à voir ....
        U1+=[np.mean([(sig[i+j-dt_prec]-sig[i+j-dt_prec-1])for j in range(inter)])]
        U2+=[np.mean([(sig[i-dt_prec+1]-sig[i+j-dt_prec-1]+2*sig[i+j-dt_prec]) for j in range(inter)])]
        D1+=[np.mean([sig[i+dt_suiv]-sig[i+j+dt_suiv-1]for j in range(inter)])]
        D2+=[np.mean([(sig[i+dt_suiv+1]-sig[i+j+dt_suiv-1]+2*sig[i+j+dt_suiv]) for j in range(inter)])]
        #D1+=[sig[i+dt_suiv]-sig[i+dt_suiv-1]] #dérivée première juste après
        #E1+=[sig[i]**2-(sig[i-dt_energy]*sig[i+dt_energy])] #Energie
        E1+=[np.mean([sig[i+j]**2-(sig[i+j-dt_energy]*sig[i+j+dt_energy]) for j in range(inter)])]
    #P=vect_detect_pic(name_sig,chemin+"A'2-A'1_300s.txt",T,opt='ripples',fact=3,max_fact=10,h=1)[dt,len(sig)-dt] #to compare with out our own selection of sharpw    
    return E1,U1,U2,D1,D2,sig[dt:len(sig)-dt],dt

#Print the result 
#print(descript(char_B))

def plot_descript(name_sig,T):
    """plot the result of the values calculated in descript"""

    E,U,U2,D,D2,sig,dt=descript(name_sig)[0]
    sig=descript(name_sig)[3]
    fig, axs = plt.subplots(4, 1)
    fig.canvas.set_window_title("Result_descript"+name_sig[66:-4])
    print(len(T[dt:len(sig)-dt]),len(E))
    axs[3].plot(T[dt:len(sig)-dt],E)
    #axs[3].set_title('Evolution de Energie')
    axs[3].set_xlabel('temps en seconde')
    axs[3].set_ylabel('E')
    fig.suptitle('Evolution des critères', fontsize=16)
    
    axs[1].plot(T[dt:len(sig)-dt],U,label="inter"+str(round(dt/512*1000,2))+"ms")
  #  axs[1].set_title('Evolution de la dérivée première avant')
    axs[1].set_xlabel('temps en seconde')
    axs[1].set_ylabel('U')
    #axs[1].legend(loc=2)
    axs[2].plot(T[dt:len(sig)-dt],D,label="inter"+str(round(dt/512*1000,2))+"ms")
   # axs[2].set_title('Evolution de la dérivée première après')
    axs[2].set_xlabel('temps en seconde')
    axs[2].set_ylabel('D')
    #axs[2].legend(loc=2)
    axs[0].plot(T[dt:len(sig)-dt],sig[dt:len(sig)-dt])
    axs[0].set_title('Signal filtré')
    axs[0].set_xlabel('temps en seconde')
    axs[0].set_ylabel('')
    #axs[0].legend(loc=2)
    plt.show()
    
def cluster(name_sig,T):
    #sig_high=filtre(name_sig,T,'ripples')
    #sig_low=filtre(name_sig,T,'epileptic')
    #E_high,U_high,U2_high,D_high,D2_high,sig_high1,dt_high=descript(sig_high)
    #E_low,U_low,U2_low,D_low,D2_low,sig_low1,dt_low=descript(sig_low)
    D_high,sig_high,sig_low,t_max_pic_high,ind,A_high,A_low=ampl_event(name_sig,T)
    Ncluster=2

    #X=np.array([E_low,U_low,U2_low,D_low,D2_low,E_high,U_high,U2_high,D_high,D2_high])
    X=np.array([D_high,A_high,A_low])#variables quantitatives étudiées
    X=X.T
    #X=X.t_max_pic_high
    kmeans = KMeans(n_clusters=Ncluster,n_init=200).fit(X)
    print(kmeans.cluster_centers_)
    predictions=kmeans.predict(X)
    #print(predictions)
    for i in range(Ncluster):
        if i in predictions :
           # example_i=where(predictions==i)[0][0]
            print(i)
    #print(len(sig_high),len(sig_low))
    #print(len(T[dt_high:len(sig_high)-dt_high]),len(sig_high1))
    fig, axs = plt.subplots(3, 1)
    fig.canvas.set_window_title("Cluster"+name_sig[66:-4])
    fig.suptitle('Clustering', fontsize=16)
    # axs[0].plot(T[dt_high:len(sig_high)-dt_high],sig_high1,label="120-250 Hz")
    # axs[0].set_xlabel('')
    # axs[0].set_ylabel('')
    # axs[0].legend(loc=2)
    # axs[1].plot(T[dt_low:len(sig_high)-dt_low],sig_low1,label="10-80 Hz")
    # axs[1].legend(loc=2)
    # axs[2].set_title('Attributions des groupes')
    # axs[2].plot(T[dt_high:len(sig_high)-dt_high],predictions)
    
    axs[0].plot(T,sig_high,label="120-250 Hz")
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].legend(loc=2)
    axs[1].plot(T,sig_low,label="10-80 Hz")
    axs[1].legend(loc=2)
    axs[2].set_title('Attributions des groupes')
    axs[0].plot(t_max_pic_high,predictions,'r.')
    axs[2].plot(t_max_pic_high,predictions,'r.')
    plt.show()
    
  
    #         peak_ind_example=all_peak_ind[example_i]
    #         marge=int(100*ms*fs)
    #         m,M=min(sig[peak_ind_example-marge:peak_ind_example+marge]),max(sig[peak_ind_example-marge:peak_ind_example+marge])
    #         figure()
    #         plot(array(range(2*marge))/fs*1000,sig[peak_ind_example-marge:peak_ind_example+marge])
    #         plot([marge/fs*1000,marge/fs*1000],[m,M],'r--')
    #         title('Example cluster '+str(i))
    #         xlabel('Temps(ms)')
    #         ylabel('LFP (microvolt)')
    return predictions,D_high,A_high,A_low
##Nouvelle idée : On passe de l'analyse de point à l'analyse de d'intervalle de 20 ms avec un pas de 10 ms entre les intervalles

def sig_to_intersig(X):
    """Signal qui prend en entrée un signal X et qui renvoie une liste de liste contenant les des intervalles de 20 ms soit 10 points""" 
    n=len(X)
    I=[]#list de list
    T_inter=[]
    sig=[]
    for i in range(5,n-5,5):
        I+=[X[i-5:i+5]]
        T_inter+=[i/512]
        sig+=[X[i]]
    return I,T ,sig
    
def ampl_event(name_sig,T):
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
            A_low+=[max(l_low)-min(l_low)]
        else:
            del ind[i]
            del D_high[i]
            del t_max_pic_high[i]
            taille_ind-=1
        i+=1
    return D_high,sig_high,sig_low,t_max_pic_high,ind,A_high,A_low
    
def duree_event(name_sig,T):
    """Donne la duree de l'evenement soir la largeur de l'enveloppe"""
    sig_high=filtre(name_sig,T,'ripples')
    sig_low=filtre(name_sig,T,'epileptic')
    t_max_pic_high,p_max_high,inter_pic_high,int_high,ind=detec_pic(name_sig,T,'ripples',2,70,1) #Critère detection bas
    #t_max_pic_low,p_max_low,inter_pic_low,int_low=detec_pic(name_sig,T,'epileptic',1,50,1) #Critère detection bas
    
    D_high=[len(elem)/512 for elem in int_high]
    return D_high,sig_high,sig_low,t_max_pic_high,ind
 

     
  
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
   
    # axs[0].xlabel("Groupe")
    # axs[1].xlabel("Groupe")
    # axs[2].xlabel("Groupe")
    #plt.ylabel("Duree de l'evenement")
    plt.show()

def description_inter(X):
    """"Donne un vecteur Y contenant les variables qualitative qui vont décrire nos intervalles définit plus tôt"""
    Inter,T,sig=sig_to_intersig(X)
    U=[] #dérivée première
    U2=[] #dérivée deuxième
    P=[] #energie
    
    n=len(Inter)
    for i in range(n): #On parcourt l'ensembles des intervalles
        U+=[np.mean([(Inter[i][j-1]-Inter[i][j])for j in range(1,Inter[i])])]
        U2+=[np.mean([(Inter[i][j+1]-Inter[i][j-1]+2*Inter[i][j]) for j in range(1,iInter[i]-1)])]
        P+=[norm(Inter[i])] # puissance signal
    return U,U2,P,T,sig
    
def plot_descript_inter(X):
    """plot les variables descritpives utilisees"""
    U,U2,P,T,sig=description_inter(X)
    fig, axs = plt.subplots(4, 1)
    fig.canvas.set_window_title("Description variables pré cluster "+name_sig[66:-4])
    fig.suptitle('Var_Descript_Inter', fontsize=16)
    axs[0].plot(T,sig,label="120-250 Hz")
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].legend(loc=2)
    axs[1].plot(T,U,label="Moyenne Dérivée première")
    axs[1].legend(loc=2)
    axs[2].set_title('Moyenne Dérivée seconde')
    axs[2].plot(T,U2)
    axs[0].plot(T,P,'r.',label="Puissance ")
    axs[3].legend(loc=2)
    axs[3].set_xlabel('Temps en seconde')
    plt.show()


##Etudions la pertinence des critères utilisés    
    
##Analysons les critères observées

#plot_descript(char_B,T)
plot_influ_crit(char_B,T)
## Clustering
#cluster(char_B,T)
# sort_sharpw_ripples(char_B,T,'ripples',1)
## Sur des evenements sélectionnés avec un critère très faible : 
#print(duree_event(char_B,T))
