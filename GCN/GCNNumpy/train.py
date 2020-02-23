import numpy as np
import pandas as pd

#activation function
def relu(x):
    s = 1 / (1 + np.exp(-x))
    return s

#adjacency matrix of directed graph
A=np.matrix([
    [0,1,0,0],
    [0,0,1,1],
    [0,1,0,0],
    [1,0,1,0]],
    dtype=float
)
#print("A = \n", A)

#feature matrix
X = np.matrix([[i,-i] for i in range(A.shape[0])], dtype=float)
#print("X = \n", X)

#self loop
I = np.matrix(np.eye(A.shape[0]))
#print("I=\n", I)

#adjacency matrix with self loop
A_hat = A + I
#print("A_hat = \n", A_hat)

#degree matrix
D=np.array(np.sum(A,axis=0))[0]
D=np.matrix(np.diag(D))
#print("D = \n", D)

#degree matrix of A_hat which has self loop
D_hat=np.array(np.sum(A_hat,axis=0))[0]
D_hat=np.matrix(np.diag(D_hat))
#print("D_hat = \n", D_hat)

#weight matrix
W = np.matrix([
    [1,-1,2],
    [-1,1,-2]
])
#print("W = \n", W)


print("A_hat = \n", A_hat)

#normalize the row of A_hat only
print("D_hat**-1 * A_hat = \n", D_hat**-1 * A_hat)


print("D_hat**-1 * A_hat * X = \n", D_hat**-1 * A_hat * X)


print("D_hat**-1 * A_hat * X * W = \n", D_hat**-1 * A_hat * X * W)

#the output is the first hiden layer
print("H=relu(D_hat**-1 * A_hat * X * W) = \n", relu(D_hat**-1 * A_hat * X * W))





