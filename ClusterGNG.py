# Growing Neural Gas + clustering

 # les imports
 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import pandas as pd
from ast import literal_eval
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import spectral_clustering
import matplotlib.animation as animation
from IPython.display import HTML
from neupy import algorithms
from scipy.interpolate import UnivariateSpline
from sklearn.metrics.pairwise import cosine_similarity
from extract_features import *

#from tqdm import tqdm_notebook as tqdm

# x,y = make_blobs(n_samples=1000, n_features=2,centers=2,cluster_std=0.4)
# # x2,y2 = make_blobs(n_samples=1000, n_features=2,centers=2,cluster_std=0.4)
# plt.scatter(*x.T,alpha=0.2)
# plt.show()
# #plt.plot(x2,'r+')
# plt.show()
# x4=np.zeros((1000,4))
# x4[:,0:2]=x
# x4[:,2:4]=x2
# neural_gas = algorithms.competitive.growing_neural_gas.GrowingNeuralGas(n_inputs=2,shuffle_data=True,verbose=True, max_edge_age=10, n_iter_before_neuron_added=50,max_nodes=100)
# # # plt.figure()
# # # plt.plot(x,'b+')
# # 
# neural_gas.train(x,epochs=10)


def draw_edge(node_1, node_2, alpha=1.):
    weights = np.concatenate([node_1.weight, node_2.weight])
    return plt.plot(*weights.T, color='black', zorder=500, alpha=alpha)

def draw_graph(graph, alpha=1.):
    for node_1, node_2 in graph.edges:
        draw_edge(node_1, node_2, alpha)
        
def GNG2ntxgraph(graph):
    """ convert a GNG graph to a network kx graph
    On considère que tous nos noeuds sont numéroté de 1 à n avec pour numérotation leur indice dans la liste des noeuds"""
    G = nx.Graph()
    i=0
    for elem in neural_gas.graph.edges : # a chaque itération on considère l'arète entre le noeud i et i+1
        #print(elem[0].weight)
        l_coordo=[nodes[1] for nodes in G.nodes(data='coordo')]
        if elem[0].weight not in list(l_coordo):
            if elem[1].weight not in list(l_coordo): # On étudie si l'un des noeud a déjà été ajouté
                G.add_node(i,coordo=elem[0].weight)
                G.add_node(i+1,coordo=elem[1].weight)
                G.add_edge(i,i+1)

            else:
                G.add_node(i,coordo=elem[1].weight)            
                G.add_edge(i,l_coordo.index(elem[1].weight)) #l'elem qui 

        else :
            if elem[1].weight not in list(l_coordo): 
                G.add_node(i+1,coordo=elem[1].weight)  
                G.add_edge(i+1,l_coordo.index(elem[0].weight))
        i=i+1    
    return G
    
    
def similarity(f1,f2):
    """Etant donné deux noeuds/ deux features on donne leur similarité, on utilise la cosine similarity""" 
    x1=np.array(f1)
    x2=np.array(f2)
    x1=x1.reshape(1,-1) # necessaire sinon erreur ....
    x2=x2.reshape(1,-1)
    return cosine_similarity(x1,x2)[0][0]

def weight_mat(graph):
    """ Créer une matrice contenant comme poids entre les arètes, leur cosine similarity """
    #On créer une liste de liste contenant tous les features
    list_node=[list(elem.weight[0]) for elem in neural_gas.graph.nodes]
    #print(list_node)
    i=0
    n=len(list_node)
    W=np.zeros((n,n))
    #print('list_node',list_node)
    for node in list_node:
        set_edge=neural_gas.graph.edges_per_node[neural_gas.graph.nodes[i]] # retourne un set contenant les voisins de node
        #print('set',set_edge)
        
        # Tant que le set n'est pas vide
        while set_edge!=set() : 
            connected_node=set_edge.pop().weight[0]
            
            #print('connected_node',connected_node)
            index_node=list_node.index(list(connected_node)) # donne l'indice du noeud dans le set
            W[i,index_node]=similarity(node,connected_node)
            #print(i,index_node)
            #print(similarity(node,connected_node))
            #print(W[i,index_node])
           # print(W)
        #print(W)
        i=i+1
    #print(W)
    return W
        #on étudie ses voisins    
def mat_D(W):
    """retourne une matrice diagonale, qui pour (i,i) contient la somme des point du noeud i avec tous ses voisins"""
    n=W.shape[0]
    D=np.zeros((n,n))
    for i in range(n):
        #print(np.reshape(W[i,:],(1,n)).shape)
        D[i,i]=np.dot(W[i,:],np.ones((n,1)))
    return D

def repartition(L):
    """ Etant donnée la matrice Laplacian D-W retourne la deuxième plus petite valeur propre ainsi que el vecteur propre associé, parttionne l'arbre en deux """
    val,vect=np.linalg.eig(L)
    vals=np.sort(val)
    value=vals[1] #deuxième plus petit element
    index_value=np.where(val==value)[0][0] # l'indice du deuxième plus petit
    vecteur=vect[index_value,:]
    return value,vecteur

def plot_vect(vecteur):
    """ indique les différentes caractéristiques du vecteur"""
    plt.figure()
    plt.boxplot(vecteur)
    plt.title('Répartition des valeurs dans le vecteur')
    print('Moyenne',np.mean(vecteur))
    print('Mediane',np.median(vecteur))
    plt.show()

def calc_b(vectA,vectB):
    return np.sum(vectA)/np.sum(vectB)

def bipartie(vecteur,T=0):
    """Afin de séparer l'arbre en deux on attribut deux valeurs étant donnée le seuil T"""
    vectA=[] # au dessus de T
    indvectA=[]
    vectB=[] # en dessous de T
    indvectB=[]
    for i in range(len(vecteur)):
        if vecteur[i]>=T:
            vectA+=[vecteur[i]]
            indvectA+=[i]
        else:
            vectB+=[vecteur[i]]
            indvectB+=[i]
    return vectA,indvectA,vectB,indvectB
    
def continue2discrete(vecteur,b,indvectA,indvectB):
    """ Passage du monde continue à un vecteur discret de 1 et -b"""
    n=len(vecteur)
    vect_rep=np.zeros((n,1))
    for i in range(n):
        if i in indvectA:
            vect_rep[i,0]=1
        else:
            vect_rep[i,0]=-b
    return vect_rep


        
def draw_scatter(nodes,dim1=0,dim2=1):
    list_nodex=[elem.weight[0][dim1] for elem in nodes] #première dim
    list_nodey=[elem.weight[0][dim2] for elem in nodes] #deuxième dim
    plt.figure(figsize=(15,30))
    plt.scatter(list_nodex,list_nodey)
    plt.show()
    
def draw_scatter_groupe(nodes,vect_ind,col,dim1=0,dim2=1):
    """ Connaissant l'indice des points intéressant on les trace avec la couleur col """
    lx=[]
    ly=[]
    for elem in vect_ind:
        lx+=[nodes[elem].weight[0][dim1]]
        ly+=[nodes[elem].weight[0][dim2]]
    plt.scatter(lx,ly,c=col)

def draw_scatter3D_groupe(ax,nodes,vect_ind,col,dim1=0,dim2=1,dim3=2) : 
    lx=[]
    ly=[]
    lz=[]
    for elem in vect_ind:
        lx+=[nodes[elem].weight[0][dim1]]
        ly+=[nodes[elem].weight[0][dim2]]
        lz+=[nodes[elem].weight[0][dim3]]
    ax.scatter(lx,ly,lz,c=col)
    return True
    
def draw_edges(node_1,node_2,dim1=0,dim2=1,alpha=0.5):
    """Trace les connections des points projeté dans dim1 et dim2"""
    lx=[node_1.weight[0][dim1],node_2.weight[0][dim1]]
    ly=[node_1.weight[0][dim2],node_2.weight[0][dim2]]
    plt.plot(lx,ly,color='black',alpha=alpha)

def draw_edges3D(ax,node_1,node_2,dim1=0,dim2=1,dim3=2,alpha=0.5):
    """Trace les connections des points projeté dans dim1,dim2 et dim3"""
    lx=[node_1.weight[0][dim1],node_2.weight[0][dim1]]
    ly=[node_1.weight[0][dim2],node_2.weight[0][dim2]]
    lz=[node_1.weight[0][dim3],node_2.weight[0][dim3]]
    ax.plot3D(lx,ly,lz,color='black',alpha=alpha)
    
def plot_repartition(graph,alpha,dim1,dim2,vectindA,vectindB):
    plt.figure(figsize=(15,30))
    for node_1, node_2 in graph.edges:
        draw_edges(node_1,node_2,dim1,dim2,alpha=0.5)
    draw_scatter_groupe(graph.nodes,vectindA,'red',dim1,dim2)
    draw_scatter_groupe(graph.nodes,vectindB,'blue',dim1,dim2)
    plt.show()
    
def plot3D_repartition(graph,alpha,dim1,dim2,dim3,vectindA,vectindB):
    """ Affiche en 3D"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for node_1, node_2 in graph.edges:
        draw_edges3D(ax,node_1,node_2,dim1,dim2,dim3,alpha=0.5)
    draw_scatter3D_groupe(ax,graph.nodes,vectindA,'red',dim1,dim2,dim3)
    draw_scatter3D_groupe(ax,graph.nodes,vectindB,'blue',dim1,dim2,dim3)
    plt.show()
    return True


    
def analyse_groupe(graph,vectindA,vectindB,tmax):
    """ Création d'un df avec afin d'analyser le comportement de chaque groupe on ajoute classe et temps du pics"""
    data=np.zeros((len(vectindA)+len(vectindB),5)) #na ligne 5 colonnes pour 4 features
    #dataB=np.array((len(vectindB),5))
    for elem in vectindA:
       # print(data)
        #print(elem)
        #print(data[elem,0:4])
        #print(graph.nodes[elem].weight[0].shape)
        data[elem,0:4]=list(graph.nodes[elem].weight[0])
        data[elem,4]=1
        # data[elem,5]=tmax[elem]
        # ia+=1
    # ib=0
    for elemb in vectindB:
        data[elemb,0:4]=list(graph.nodes[elemb].weight[0])
        data[elemb,4]=2
        # data[elemb,5]=tmax[elem]
        
        # ib+=1
    df=pd.DataFrame(data,index=[i for i in range(len(vectindA)+len(vectindB))],columns=['nb_oscill','freq','p_max_high','A_high','classe'])
    print(df.describe(include='all'))
    print(df.loc[df['classe']==2,:])
    print(df.loc[df['classe']==1,:].describe())
    return df
    
def groupe2pic(tpic,datagr):
    """ gives the time of the pic for each pic"""
    return true 
        
    # W=[elem.weight for elem in neural_gas.graph.nodes]
#A=nx.adjacency_matrix(GNG2ntxgraph(neural_gas.graph))
#labels=sklearn.cluster.spectral_clustering(A)

## Sur nos données : 

# on importe nos données 
# 
# from extract_features import *
# 
# time=1800
# T=[round(i/512,6) for i in range(1,time*512+1)]
# chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/'
# char_B =chemin+"B'2-B'1_1800s.txt"
# 
#t_max_pic_high,p_max_high,inter_pic_high,int_high,ind,A_high,sig_high,nb_oscill,freq=crit_clust(char_B,T)
# data=np.array([p_max_high,freq]).transpose()
# plt.figure()
# plt.xlabel('Puissance')
# plt.ylabel('Frequence')
# plt.scatter(data[:,0],data[:,1],marker='o',s=0.1)
# 
# neural_gas = algorithms.competitive.growing_neural_gas.GrowingNeuralGas(n_inputs=2,shuffle_data=True,verbose=True, max_edge_age=10, n_iter_before_neuron_added=50,max_nodes=100)
# 
# neural_gas.train(data,epochs=100)
# # now restore stdout function
# 
# plt.figure()
# plt.xlabel('Puissance')
# plt.ylabel('Frequence')
# draw_graph(neural_gas.graph)
# plt.show()

## Pour 4 features
time=1800
T=[round(i/512,6) for i in range(1,time*512+1)]
chemin ='/Users/iris/Desktop/Projet_Rech/Exemple/EEG_58_Sig/Donnes_signaux/'
char_B =chemin+"B'2-B'1_1800s.txt"

t_max_pic_high,p_max_high,inter_pic_high,int_high,ind,A_high,sig_high,nb_oscill,freq=crit_clust(char_B,T)

data=np.array([nb_oscill,freq,p_max_high,A_high]).transpose()
neural_gas = algorithms.competitive.growing_neural_gas.GrowingNeuralGas(n_inputs=4,shuffle_data=True,verbose=True, max_edge_age=10, n_iter_before_neuron_added=50,max_nodes=100)
# 
neural_gas.train(data,epochs=100)
W=weight_mat(neural_gas.graph)
D=mat_D(W)
val,vect=repartition(D-W)
plot_vect(vect)
# Grâce à l'affichage on considère que 0 est bon seuil
vectA,indvectA,vectB,indvectB=bipartie(vect)
b=calc_b(vectA,vectB)
vect_rep=continue2discrete(vect,b,indvectA,indvectB)
df=analyse_groupe(neural_gas.graph,indvectA,indvectB,t_max_pic_high)
# draw_graph(neural_gas.graph)
# plt.show()
# 
# # Matrice d'adjacence du graphe : 
# A=nx.adjacency_matrix(neural_gas.graph.nodes)
# labels=sklearn.cluster.spectral_clustering(A)
