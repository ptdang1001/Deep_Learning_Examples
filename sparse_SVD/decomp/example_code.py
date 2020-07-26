import numpy as np
import pandas as pd

from SparseMat import SparseMat
from svd_decomp import svd_decomp


def getMatrixOutput(data):
    ACCURACY = 3  # this does not affect any data, just how it's printed

    dim = data.dim
    output = np.zeros((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            entry = round(value_at(data, i, j), ACCURACY)
            output[i][j] = entry
    return (output)


def value_at(data, i, j):
    for col in range(data.row_index[i], data.row_index[i + 1]):
        if data.col_index[col] == j:
            return data.value[col]
    return 0


#generates a random sparse matrix with given "density"
#this is not used in any algorithm but helpful to test and debug the code.
def rand_mat(n, m, density):

    data = np.random.rand(n * m)
    for i in range(1, len(data)):
        if (np.random.rand() > density):
            data[i] = 0

    M = []
    for i in range(0, n):
        row = []
        for j in range(0, m):
            row.append(data[(m * i) + j])
        M.append(row)

    #now do the sparse implementation
    value = []
    col_index = []
    row_index = []

    row_index.append(0)
    for i in range(0, n):
        for j in range(0, m):
            if M[i][j] != 0:
                value.append(M[i][j])
                col_index.append(j)
        row_index.append(len(col_index))

    return SparseMat(value, col_index, row_index)


#generates a random sparse matrix with given "density"
#this is not used in any algorithm but helpful to test and debug the code.
def preProcessMyData(data):
    M = []
    n, m = data.shape
    data = data.flatten()
    for i in range(0, n):
        row = []
        for j in range(0, m):
            row.append(data[(m * i) + j])
        M.append(row)

    #now do the sparse implementation
    value = []
    col_index = []
    row_index = []

    row_index.append(0)
    for i in range(0, n):
        for j in range(0, m):
            if M[i][j] != 0:
                value.append(M[i][j])
                col_index.append(j)
        row_index.append(len(col_index))

    return SparseMat(value, col_index, row_index)


#here we generate a random 30x30 matrix and compute the SVD.
def main():
    n_row = 6
    n_col = 6
    density = 0.05
    #M = rand_mat(n_row, n_col, density)
    myData = np.random.rand(n_row, n_col)
    n, m = myData.shape
    for i in range(n):
        for j in range(n):
            if (np.random.rand() > density):
                myData[i][j] = 0.0

    print("mydata:")
    print(myData)
    M = preProcessMyData(myData)

    # get u, sigma, v
    u, s, v = svd_decomp(M)

    u2 = getMatrixOutput(u)
    print("u:")
    print(u2)

    s2 = getMatrixOutput(s)
    print("sigma:")
    print(s2)

    v2 = getMatrixOutput(v)
    print("v:")
    print(v2)

    print("final:")
    print(np.dot(np.dot(u2, s2), v2))
    return ()


# end

if __name__ == '__main__':
    main()
