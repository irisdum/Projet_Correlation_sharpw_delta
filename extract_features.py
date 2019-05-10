#Extraction des features pour l'analyse des SPW-Rs

#les imports
import numpy as np
import matplotlib.pyplot as plt
from Traitement_fich import *

time=1800
T=[round(i/512,6) for i in range(1,time*512+1)]
chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/'
#Extraction de l'amplitude du SPW-Rs
char_A=chemin+"A'2-A'1_300s.txt" #a changer en fonction de la taille de l'extrait considéré
char_B =chemin+"B'2-B'1_1800s.txt"
char_O=chemin+"O'9-O'8_1800s.txt"

from scipy.fftpack import fft

# Critère 1 : l'amplitude

def crit_clust(name_sig,T):
    """ Donne l'amplitude des SPW-Rs dans le signal filtré des pics détecté dans detec pic ainsi que la puissance de ces pics, le nombre d'oscillation supérieur à 50% de l'amplitude max ainsi que la fréquence lié au pic de maximum amplitude"""
    sig_high=filtre(name_sig,T,'ripples')
    t_max_pic_high,p_max_high,inter_pic_high,int_high,ind=detec_pic(name_sig,T,'ripples',1,50,1)
    A_high=[]
    taille_ind=len(ind)
    nb_oscill=[]
    freq=[]
    i=0
    while i<taille_ind: 
        # print(ind[i])
        if ind[i][0]!=ind[i][1]:#normalement impossible que ce soit égaux ...
            # print('lhigh',sig_high[ind[i][0]:ind[i][1]])
            L_high=sig_high[ind[i][0]:ind[i][1]+1] 
            
            #print('ind',ind[i][0],ind[i][1])
            A_high+=[max(L_high)-min(L_high)]
            nb_oscill+=[nb_osc(L_high,max(L_high))] 
            #A_high+=[np.linalg.norm(l_high)]
            freq+=[freq_pic(L_high)]
        i+=1
    #print(i)
    if len(A_high)<taille_ind:
        print('pb de TAILLE')
    return t_max_pic_high,p_max_high,inter_pic_high,int_high,ind,A_high,sig_high,nb_oscill,freq
    



def nb_osc(sig,max):
    """ Ayant un signal sig et une amplitude max, on compte le nombre de signaux étant 50% au dessus de l'amplitude max"""
    m=0
    nb_oscill=0
    for i in range(1,len(sig)-1) : 
        if sig[i] > 0.5*max: # elem du signal au dessus du seuil
            if sig[i-1]<0.5*max:#l'element juste avant est en dessous du seuil
                m=1 #on est dans une des variations que l'on cherche à detecter
            if sig[i+1]<0.5*max:
                nb_oscill+=m
    return nb_oscill
 
def freq_pic(sig):
    """ Etant donné un signal sig 
    Retourne la fréquence lié au max d'amplitude"""
    n=len(sig)
    t=1/512
    yf=fft(sig)
    xf=np.linspace(0.0,1.0/(2.0*t),n//2)
    yfcorr=2.0/n*np.abs(yf[0:n//2])
    #plt.plot(xf,yfcorr)
    ind_max=np.argmax(yfcorr)
    #print(xf[ind_max])
    #plt.grid()
    #plt.show()
    return xf[ind_max]
    
    
    
def plot_crit(name_sig,T):
    """Affche les différentes critères"""
    t_max_pic_high,p_max_high,inter_pic_high,int_high,ind,A_high,sig_high,nb_oscill,freq=crit_clust(name_sig,T)

    #plt.plot(t_max_pic)
    figure = plt.figure(figsize = (30, 30))
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1,
                        right = 0.9, top = 0.85, wspace = 0, hspace = 0.3)
    axes = figure.add_subplot(4, 1, 1)
    #axes.set_xlabel('Temps en seconde')
    axes.set_ylabel('axe des y')
    axes.set_title('Signal fitré 120-250 Hz '+ name_sig[66:-4])
    axes.plot(T,sig_high, color = 'blue')
    
    axes = figure.add_subplot(4, 1, 2)
    #axes.set_xlabel('Temps en sec')
    axes.set_ylabel('Amplitude')
  #  axes.set_title('Amplitude')
    axes.plot(t_max_pic_high,A_high, color = 'red')
    
    axes = figure.add_subplot(4, 1, 3)
    axes.set_xlabel('Temps en sec')
    axes.set_ylabel('Nombre oscillations')
  #  axes.set_title('Amplitude')
    axes.plot(t_max_pic_high,nb_oscill,color = 'green')
    
    axes = figure.add_subplot(4, 1, 4)
    axes.set_xlabel('Temps en sec')
    axes.set_ylabel('Hz')
  #  axes.set_title('Amplitude')
    #print(len(freq))
    axes.plot(t_max_pic_high,freq,color = 'black')
    plt.show()
    

#plot_crit(char_B,T)

def analyse_crit(name_sig,T):
    t_max_pic_high,p_max_high,inter_pic_high,int_high,ind,A_high,sig_high,nb_oscill,freq=crit_clust(name_sig,T)
    data = [np.array(A_high),np.array(nb_oscill),np.array(freq),np.array(p_max_high)]
    figure = plt.figure(name_sig[66:-4],figsize = (30, 30))
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1,
                        right = 0.9, top = 0.85, wspace = 0, hspace = 0.3)
    
    print('On detecte'+ str(len(t_max_pic_high))+ ' pics')
    axes = figure.add_subplot(2, 2, 1)
    #axes.set_xlabel('Temps en seconde')
    axes.set_ylabel('puissance')
    axes.set_title('Puissance pic ')
    axes.boxplot(p_max_high)
    
    axes = figure.add_subplot(2, 2, 2)
    #axes.set_xlabel('Temps en sec')
    #axes.set_ylabel('Amplitude')
    axes.set_title('Amplitude')
    axes.boxplot(A_high,)
    
    axes = figure.add_subplot(2,2, 3)
    #axes.set_xlabel('Temps en sec')
    #axes.set_ylabel('Nombre oscillations')
    axes.set_title('Nombre oscillation')
    axes.boxplot(nb_oscill)
    
    axes = figure.add_subplot(2,2, 4)
    #axes.set_xlabel('Temps en sec')
    axes.set_ylabel('Hz')
    axes.set_title('Fréquence')
    #print(len(freq))
    axes.boxplot(freq)

    plt.show()

analyse_crit(char_B,T)