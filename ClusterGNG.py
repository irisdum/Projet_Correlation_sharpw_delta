# Growing Neural Gas + clustering

 # les imports
 
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.datasets import make_blobs, make_moons
import matplotlib.animation as animation
from IPython.display import HTML
from neupy import algorithms
from scipy.interpolate import UnivariateSpline
#from tqdm import tqdm_notebook as tqdm

# x,y = make_blobs(n_samples=1000, n_features=2,centers=2,cluster_std=0.4)
# x2,y2 = make_blobs(n_samples=1000, n_features=2,centers=2,cluster_std=0.4)
# plt.scatter(*x.T,alpha=0.2)
# plt.show()
# #plt.plot(x2,'r+')
# plt.show()
# x4=np.zeros((1000,4))
# x4[:,0:2]=x
# x4[:,2:4]=x2
# neural_gas = algorithms.competitive.growing_neural_gas.GrowingNeuralGas(n_inputs=2,shuffle_data=True,verbose=True, max_edge_age=10, n_iter_before_neuron_added=50,max_nodes=100)
# # plt.figure()
# # plt.plot(x,'b+')
# 
# neural_gas.train(x,epochs=100)

def draw_edge(node_1, node_2, alpha=1.):
    weights = np.concatenate([node_1.weight, node_2.weight])
    return plt.plot(*weights.T, color='black', zorder=500, alpha=alpha)

def draw_graph(graph, alpha=1.):
    for node_1, node_2 in graph.edges:
        draw_edge(node_1, node_2, alpha)


# plt.figure()
# draw_graph(neural_gas.graph)
# plt.show()
# grahic_node=algorithms.competitive.growing_neural_gas.NeuronNode()
# print(neural_gas.graph.nodes)
# Node=algorithms.competitive.growing_neural_gas.NeuronNode
# 
# data, _ = make_moons(10000, noise=0.06, random_state=0)
# plt.scatter(*data.T)

# 
# utils.reproducible()
# gng = algorithms.GrowingNeuralGas(
#     n_inputs=2,
#     n_start_nodes=2,
# 
#     shuffle_data=True,
#     verbose=False,
#     
#     step=0.1,
#     neighbour_step=0.001,
#     
#     max_edge_age=50,
#     max_nodes=100,
#     
#     n_iter_before_neuron_added=100,
#     after_split_error_decay_rate=0.5,
#     error_decay_rate=0.995,
#     min_distance_for_update=0.2,
# )
# # 
# fig = plt.figure()
# plt.scatter(*data.T, alpha=0.02)
# plt.xticks([], [])
# plt.yticks([], [])
# fig = plt.figure()
# plt.scatter(*data.T, alpha=0.02)
# plt.xticks([], [])
# plt.yticks([], [])

def animate(i):
    for line in animate.prev_lines:
        line.remove()
        
    # Training will slow down overtime and we increase number
    # of data samples for training
    n = int(0.5 * gng.n_iter_before_neuron_added * (1 + i // 100))
    
    sampled_data_ids = np.random.choice(len(data), n)
    sampled_data = data[sampled_data_ids, :]
    gng.train(sampled_data, epochs=1)
    
    lines = []
    for node_1, node_2 in gng.graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        line, = plt.plot(*weights.T, color='black')

        plt.setp(line, linewidth=1, color='black')
        
        lines.append(line)
        lines.append(plt.scatter(*weights.T, color='black', s=10))
    
    animate.prev_lines = lines
    return lines

# animate.prev_lines = []
# anim = animation.FuncAnimation(fig, animate, tqdm(np.arange(220)), interval=30, blit=True)
# HTML(anim.to_html5_video())
# neural_gas.graph.n_nodes
# print(len(neural_gas.graph.edges))
# neuron_1, neuron_2 = edges[0]
# neuron_1.weight
# neuron_2.weight