import numpy as np

#activation funtion relu f(x)= 1/(1+e^-x)
def relu(x):
    s = 1 / (1 + np.exp(-x))
    return s

#one layer of GCN
'''
A_hat:adjacency matrix which adds self loop(considering the feature of itself)
D_hat:degree matrix of A_hat
X:feature matrix (Embeding H0)
W:weight matrix
'''
def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)

