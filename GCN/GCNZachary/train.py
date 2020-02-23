import numpy as np
import networkx as nx
from networkx import to_numpy_matrix
import matplotlib.pyplot as plt


import modules
import tools

#data
zkc=nx.karate_club_graph()

#order
order = sorted(list(zkc.nodes()))

#adjacency matrix
A = to_numpy_matrix(zkc, nodelist=order)

#adding self loop
I = np.eye(zkc.number_of_nodes())
A_hat = A + I

#degree
D_hat=np.array(np.sum(A_hat, axis=0))[0]
D_hat=np.matrix(np.diag(D_hat))

#weight1 matrix
W_1 = np.random.normal(loc = 0, scale=1, size=(zkc.number_of_nodes(), 4))

#weight2 matrix
W_2 = np.random.normal(loc = 0, size=(W_1.shape[1], 2))

#hidden 1
H_1 = modules.gcn_layer(A_hat, D_hat, I, W_1)

#hidden 2
H_2 = modules.gcn_layer(A_hat, D_hat, H_1, W_2)

#output
output = H_2


feature_representations = {node: np.array(output)[node] for node in zkc.nodes()}

#draw the graph of original data
tools.plot_graph(zkc)

#
plt.figure()
for i in range (34):
    if zkc.nodes[i]['club'] == 'Mr. Hi':
        plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,color = 'b',alpha=0.5,s = 100)
    else:
        plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,color = 'r',alpha=0.5,s = 100)


H = nx.Graph()

node_num = len(feature_representations)

nodes = list(range(node_num))  # 34 nodes

# add edges
for i in range(node_num):
    src = i
    for j in range(node_num):
        if A[i, j] != 0 and i != j:
            dest = j
            H.add_edge(src, dest)

nx.draw_networkx_edges(H, feature_representations, alpha=0.3)
plt.show()



