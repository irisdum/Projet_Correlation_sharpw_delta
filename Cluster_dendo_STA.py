import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from Traitement_fich import*
from extract_features import*
from STA import*
import ast
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth # for kdtree method
from scipy.cluster.hierarchy import*
from scipy.spatial.distance import pdist
char_A=chemin+"A'2-A'1_300s.txt" #a changer en fonction de la taille de l'extrait considéré
char_B =chemin+"B'2-B'1_1800s.txt"
char_O=chemin+"O'9-O'8_1800s.txt"
# Environ deux secondes à 4 secondes séparent des SPW

#On regarde sur les deux secondes avant en faisant des intervalles de 10 ms
delta=1024
# size_inter=round(0.01*512,0)
size_inter=1
time=1800
# POur le cas où on regarde juste le nombre de rythme delat détecté
T=[round(i/512,6) for i in range(1,time*512+1)]


def asc_cluster(name_sig,T,method='ward',disp=True,nb_gr=2):
    """ Applique une méthode ascendante nos données"""
    #nb_gr=4 #we choose two groups
    # D_high,sig_high,sig_low,t_max_pic_high,ind,A_high,A_low=crit_event(name_sig,T)    
    t_max_pic_high,p_max_high,inter_pic_high,int_high,ind,A_high,sig_high,nb_oscill,freq=crit_clust(name_sig,T)
    #print(t_max_pic_high)
    X=np.array([p_max_high,A_high,nb_oscill,freq])#quantitatives variable
    # X=np.array([D_high,A_high])
    X=X.T
    #print(np.size(X))
    Z = linkage(X, method)
    #Have to be close to 1 : 
    c, coph_dists = cophenet(Z, pdist(X))
    print('cophenecy',c)
    cluster=fcluster(Z,nb_gr,criterion='maxclust')
    dist=maxdists(Z)
    if disp:
        # 
        # plt.scatter(X[:,1],X[:,2])
        # plt.show()
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()
        plt.figure(figsize=(10, 8))
        # inconsistence=inconsistent(Z)
        # bool=is_valid_linkage(Z)
    
        plt.scatter(X[:,0], X[:,1], c=cluster)  # plot points with cluster dependent colors
        plt.title('pmax(A)')
        plt.show()
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:,1], X[:,2], c=cluster)  # plot points with cluster dependent colors
        plt.title('A(nb_oscill)')
        plt.show()
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:,2], X[:,3], c=cluster)  # plot points with cluster dependent colors
        plt.title('nb_oscill(freq)')
        plt.show()
    
    return cluster,t_max_pic_high,ind,A_high,p_max_high,nb_oscill,freq
    
def plot_influ_crit_asccluster(name_sig,T,method='ward',disp=True,nb_br=4):
    pred,t_max_pic_high,ind,A_high,p_max_high,nb_oscill,freq=asc_cluster(name_sig,T,method,disp=True,nb_gr=nb_br)
    #print('cluster',len(cluster),cluster[0])
    plt.figure()
    col=['r.','b.','g.','r.']
    fig, axs = plt.subplots(1, 4)
    fig.canvas.set_window_title("Cluster_influence_crit"+name_sig[66:-4])
    fig.suptitle('Influence criteres classification', fontsize=16)
    #print(len(D),len(pred))
    
    for i in range(len(pred)):
        axs[0].plot(pred[i],p_max_high[i],col[pred[i]])
        axs[1].plot(pred[i],A_high[i],col[pred[i]])
        axs[2].plot(pred[i],freq[i],col[pred[i]])
        axs[3].plot(pred[i],nb_oscill[i],col[pred[i]])
   
    #axs[0].set_xlabel("Groupe")
    axs[0].set_title("puissance 120-250 Hz")
    #axs[1].set_xlabel("Groupe")
    axs[1].set_title("Amplitude 120-250 Hz")
    #axs[2].set_xlabel("Groupe")
    axs[2].set_title("freq Hz")
    axs[3].set_xlabel("Groupe")
    axs[3].set_title("nombre d'oscillation'")
    #plt.ylabel("Duree de l'evenement")
    plt.show()
    
def sort_sharpw_cluster(name_sig,T,gr,nb_gr=4):
    """Donne la liste de tous les temps pour lesquel on a détecté un pic du groupe gr"""
    cluster,t_max_pic_high,ind,A_high,p_max_high,nb_oscill,freq=asc_cluster(name_sig,T,'average',False,nb_gr)
    t_max_gr=[]
    for i in range(len(t_max_pic_high)) : 
        if cluster[i]==gr:
            t_max_gr+=[t_max_pic_high[i]]
    return t_max_gr


def vect2inter(dt_fen,vect):
    """Regroupe les valeurs que l on trouve par intervalle"""
    
    global time
    Mat_interpic=[] #liste qui contiendra le nombre de pic delta pour chaque inter. 
    i=0
    # print('vect',len(vect))
    while dt_fen*i<=len(vect)-(dt_fen):
        Mat_interpic+=[sum(vect[int(dt_fen)*i:int(dt_fen)*i+int(dt_fen)])]
        i=i+1
    #print('Mat_interpic',len(Mat_interpic))
    return Mat_interpic
    
def STAparInter(list_pic_delta,dt_fen,delta):
    """liste pic_delta contient 1 lorsqu il y a une activité delta détecté,delta la fenêtre du STA"""
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
    i=0
    for elem in liste_pic_delta:
        plot_delta=vect2inter(dt_fen,elem[0,:])
        plt.plot([i for i in range(len(plot_delta))],plot_delta)
        i=i+1
    plt.title('Vecteur indicateur par intervalle')
    plt.xlabel('Numero intervalle')
    plt.ylabel('Présence de pic')
    plt.show()
    
def plot_STA(char_B,delta,opt,nbgr,gr):
    """Affichage propre STA, opt signifie before ou after, nbgr number of groups in the clustering SPW-Rs, """
    
    liste_t=sort_sharpw_cluster(char_B,T,gr,nbgr)
    vect_1=txt2STApic(char_O) #pic min fact=1, maxfact=50, h=1
    size=len(vect_1)
    X1=np.zeros((1,size))
    X1[0,:]=vect_1
    if opt=='after' : 
        x=[i for i in range(delta)]
        Liste_val_mean,Liste_std,Liste_pic_delta=STA_dt_after(liste_t,X1,delta)
    if opt=='before' : 
        Liste_val_mean,Liste_std,Liste_pic_delta=STA_dt_before(liste_t,X1,delta)
        x=[-i for i in reversed(range(delta))]
    #print(Liste_val_mean)
    fig=plt.figure("STA delta=" +str(delta)+str(opt)+" nbgr="+str(nbgr)+" gr="+str(gr))
    fig.suptitle('STA '+str(opt)+" gr = "+str(gr)+"/"+str(nbgr), fontsize=16)
    plt.plot(x,Liste_val_mean[0,:])
    #axs[0].set_title("Duree 120-250 Hz")
    plt.show()
    
    
## Test
#liste_t=sort_sharpw_cluster(char_B,T,1,4)
#plot_influ_crit_asccluster(char_B,T,method='ward',disp=True,nb_br=2)
# vect_1=txt2STApic(char_O) #pic min fact=1, maxfact=50, h=1   
# size=len(vect_1)
# X1=np.zeros((1,size))
# X1[0,:]=vect_1
# Liste_val_mean,Liste_std,Liste_pic_delta=STA_dt_after(liste_t,X1,delta)
# plot_STA(char_B,delta,'after',4,1)
# Inter=STAparInter(Liste_pic_delta,10,delta)
# plt.plot([i for i in range(len(Inter))], Inter)
# plt.show()