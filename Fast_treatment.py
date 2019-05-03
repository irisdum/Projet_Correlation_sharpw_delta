#This document enables to fasten the analysis process : creating .txt files of the power of the signal for instance

#from Traitement_fich import*
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from ast import literal_eval
#from Traitement_fich import *
import scipy
global chemin
chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/'
charB=chemin+"B'2-B'1_1800s.txt"
charO=chemin+"O'9-O'8_1800s.txt"

time=1800
T=[round(i/512,6) for i in range(1,time*512+1)]

# Power signal

def pow2txt(puiss,Tpuiss,char,h=1) : 
    f = open(char+"_pow.txt", "x")
    f.write(char+" puissance et Puiss du signal "+str(h)+"\n")
    for i in range(len(puiss)):
        f.write(str(puiss[i])+','+str(Tpuiss[i])+'\n')
    f.close()
#         
def txt2pow(char):
    # fp=open(char+"_pow.txt","r")
    # line=fp.read()
    # sig=line.split("\n")[1:-1]
    puiss=[]
    Tpuiss=[]
    fp= open(char+"_pow.txt","r")
    line = fp.readline() #on enlève la premnière ligne
    line = fp.readline()
    cnt = 1
    while line:
        p,T=line.split(',')
        puiss+=[float(p)]
        Tpuiss+=[float(T)]
        line = fp.readline()
        cnt += 1
        # print(line)
        # print(cnt)
    fp.close()
    #print('fin de la boucle ')
    return puiss,Tpuiss
# 
# def STApic2txt(char1,T,opt='ripples',fact=3,max_fact=10,h=1): 
#     vect=vect_detect_pic_STA(char1,T,opt='ripples',fact=3,max_fact=10,h=1)
#     f = open(char+"_pic.txt", "x")
#     f.write(char+" Pic pour delta pour STA et h= "+str(h)+ +"fact min" + str(fact)+"fact max = "+ str(max_fact)+"\n")
#     for i in range(len(vect)):
#         f.write(str(vect[i])+'\n')
#     f.close()
# 
# def txt2STApic(char):
#     vect=[]
#     fp= open(char+"_pic.txt","r")
#     line = fp.readline() #on enlève la premnière ligne
#     line = fp.readline()
#     cnt = 1
#     while line:
#         vect+=[float(line)]
#         line = fp.readline()
#         cnt += 1
#         # print(line)
#         # print(cnt)
#     fp.close()
#     #print('fin de la boucle ')
#     return vect
# 
# STApic2txt(charO,T,'ripples',1,20,1)

# def pic2txt(char,fact,max_fact,h):
#     """ prend en argument le nom du fichier calcul tous les pics SPW-Rs détecté puis les mets dans un fichier txt"""
#     true_peak,true_list_max,true_list_ripples,inter_pic,list_ind=detec_pic(char1,T,'ripples',fact,max_fact,h)
#     f = open(char+"_pic.txt", "x")
#     f.write(char+"Donnees sur pic SPW ensemble pic detectés h="+str(h)+"fact min" + str(fact)+"fact max = "+ str(max_fact)+"\n")
#     for i in range(len(true_peak)):
#         f.write(str(true_peak)+','+str(true_list_max)+','+str(true_list_ripples)+','+ str(inter_pic)+','+str(list_ind)+'\n')
#     f.close()

# def txt2pic(char):
#     true_peak,true_list_max,true_list_ripples,inter_pic,list_ind=[],[],[],[],[]
#     fp= open(char+"_pic.txt","r")
#     line = fp.readline() #on enlève la premnière ligne
#     line = fp.readline()
#     cnt = 1
#     while line:
#         tp,tlm,tlr,ip,li=line.split(',')
#         true_peak+=[literal_eval(tp)]
#         true_list_max+=[literal_eval(tlm)]
#         true_list_ripples+=[literal_eval(tlr)]
#         inter_pic+=[literal_eval(ip)]
#         list_ind+=[literal_eval(li)]
#         line = fp.readline()
#         cnt += 1
#        
#     fp.close()
#     #print('fin de la boucle ')
#     return true_peak,true_list_max,true_list_ripples,inter_pic,list_ind
    
        
#Faisons un test : 
# P1,T1=calc_puiss(charB,T,h=1,opt='ripples')
# #pow2txt(P1,T1,charB,1)
# plt.figure(figsize=(15,30))
# plt.subplot(2,1,1)
# 
# plt.plot(T1[0:10],P1[0:10])
# 
# plt.subplot(2,1,2)
# puiss,Tp=txt2pow(charB)
#La fonction fonctionne

# Pour 30 minutes d'enregistrement  : 
# P1,T1=calc_puiss(charB,T,h=1,opt='ripples')
# pow2txt(P1,T1,charB,1)
# P2,T2=calc_puiss(charO,T,h=1,opt='delta')
# pow2txt(P2,T2,charO,1)

#Créeons pour le fichier _pic : 
#pic2txt(charB,3,10,1)

#Determinons la période entre SPW : 
# lt_max=detec_pic(charB,T,opt='ripples',fact=3,max_fact=10,h=20)[0] #liste contenant des temps max 
# print(lt_max)
# diff=[lt_max[i+1]-lt_max[i] for i in range((len(lt_max)-1))]
# tmoy=np.mean(diff)
# print(diff)
# print(tmoy)