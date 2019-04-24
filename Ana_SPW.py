#Analyse avec les SPW

#import
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from Traitement_fich import*
from Cluster_v1 import*
from STA import*
import ast
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth # for kdtree method
from scipy.cluster.hierarchy import*
from scipy.spatial.distance import pdist

# Environ deux secondes à 4 secondes séparent des SPW

#On regarde sur les deux secondes avant en faisant des intervalles de 10 ms
delta=512
# size_inter=round(0.01*512,0)
size_inter=5
time=1200
# POur le cas où on regarde juste le nombre de rythme delat détecté
T=[round(i/512,6) for i in range(1,time*512+1)]

def vect2inter(dt_fen,vect):
    """Regroupe les valeurs que l'on trouve par intervalle"""
    
    global time
    Mat_interpic=[] #liste qui contiendra le nombre de pic delta pour chaque inter. 
    i=0
    # print('vect',len(vect))
    while dt_fen*i<len(vect)-(dt_fen+1):
        Mat_interpic+=[sum(vect[int(dt_fen)*i:int(dt_fen)*i+int(dt_fen)])]
        i=i+1
    #print('Mat_interpic',len(Mat_interpic))
    return Mat_interpic
    
def STAparInter(list_pic_delta,dt_fen,delta):
    """liste pic_delta contient 0 lorsqu il y a une activité delta détecté,delta la fenêtre du STA"""
    taille_i=len(list_pic_delta)
    Mat_inter_pic=np.zeros((taille_i,int(delta/dt_fen)))# Matrice qui contient sur chaque ligne pour un SPW le nb de pic delta par inter
    Mean_inter_pic=[]
    i=0
    for elem in list_pic_delta: #je parcours toute les sous listes extraites qui sont constitué de 0 et de 1
        Mat_inter_pic[i,:]=vect2inter(dt_fen,elem[0,:]) #j'ajoute une ligne chaque valeur correspond au nombre de pic détecté lors de l'intervalle dt_fen
        i=i+1
    for j in range(Mat_inter_pic.shape[1]):
        Mean_inter_pic+=[np.mean(Mat_inter_pic[:,j])]
    return Mean_inter_pic
    
def plot_vectind_delta(liste_pic_delta,dt_fen):
    """ Affichage du vecteur indicateur pour rythme delta pour avec une fenêtre dt après SPW"""
    plt.figure()
    for elem in liste_pic_delta:
        plot_delta=vect2inter(dt_fen,elem[0,:])
        plt.plot([i for i in range(len(plot_delta))],plot_delta)
    plt.title('Vecteur indicateur par intervalle')
    plt.xlabel('Numero intervalle')
    plt.ylabel('Présence de pic')
    plt.show()
    
## Test

#after
vect_1=vect_detect_pic_STA(char_O,T,opt='ripples',fact=3,max_fact=10,h=1) # regarde comportement rythme delta

#print('len vec1',len(vect_1))
#print(len(vect_1),'nb pic delta')
size=len(vect_1)
# print("size",size)
X1=np.zeros((1,size))

X1[0,:]=vect_1
#X1[:,1]=vect_1
delta=512
liste_t=sort_sharpw_cluster(char_B,T,4,4) #groupe 1 avec 4 cluster

#Affichons pour chaque SPW-Rs détecté le comportement du vecteur indicateur delta


# Affichons pour chaque SPW-Rs détecté, le comportement du vecteur indicateur delta passé par la fonction vect2inter ( en réduisant par intervalle)

Liste_val_mean,Liste_std,Liste_pic_delta=STA_dt_after(liste_t,X1,delta)

plot_vectind_delta(Liste_pic_delta,size_inter)


# print(len(Liste_pic_delta))
# Inter=STAparInter(Liste_pic_delta,size_inter,delta)
# plt.plot([i for i in range(len(Inter))], Inter)
# plt.show()
