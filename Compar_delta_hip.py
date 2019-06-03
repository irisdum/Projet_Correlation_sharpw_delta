# Ce fichier a pour but de comparer les bandes delta dans l'hippocampe avec les évènements SPW-Rs au sein de l'hippocampe.
#Une attention particulière sera portée sur la phase de ces rythmes. 

## Import
from Traitement_fich import*
from Cluster_dendo_STA import*


## Variables

char_B =chemin+"B'2-B'1_1800s.txt"
char_O=chemin+"O'9-O'8_1800s.txt"

#On regarde sur les deux secondes avant en faisant des intervalles de 10 ms
delta=1024
# size_inter=round(0.01*512,0)
size_inter=1
time=1800
# POur le cas où on regarde juste le nombre de rythme delat détecté
T=[round(i/512,6) for i in range(1,time*512+1)]

## Fonctions


def delta_phase(char_B,T):
    """ Retourne un vecteur avec la phase à chaque instant t de T"""
    #Y=calc_puiss(char_O,T,1,'delta')[0]#on a choisit un pas de 1 pour la puissance
    Y=filtre(char_B,T,opt='delta')
    signal_analytic=sp.hilbert(Y)
    return np.angle(signal_analytic)
    
def plot_STA_hippo(char_B,delta,opt,nbgr,gr):
    """Affichage propre STA, pour les rythmes au sein de l'hippocampe opt signifie before ou after, nbgr number of groups in the clustering SPW-Rs, """
    liste_t=sort_sharpw_cluster(char_B,T,gr,nbgr)
#    vect_1=txt2STApic(char_O) #pic min fact=1, maxfact=50, h=1
    vect_1=delta_phase(char_B,T)
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
    print("ecart-type")
    list_val_std3=[i+j for i,j in zip(Liste_std,Liste_val_mean[0,:])] #std from the new values calculated
    list_val_std4=[np.abs(j)-np.abs(i) for i,j in zip(Liste_std,Liste_val_mean[0,:])]
    fig=plt.figure("STA delta=" +str(delta)+str(opt)+" nbgr="+str(nbgr)+" gr="+str(gr))
    fig.suptitle('STA '+str(opt)+" gr = "+str(gr)+"/"+str(nbgr), fontsize=16)
    plt.plot(x,Liste_val_mean[0,:],'blue')
    print(list_val_std4)
    plt.plot(x,list_val_std4[0],'r')
    plt.plot(x,list_val_std3[0],'r')
    #axs[0].set_title("Duree 120-250 Hz")
    plt.show()
    
## Rapides tests
#plot_STA_hippo(char_B,delta,'after',2,2)