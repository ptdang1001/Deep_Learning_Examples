import numpy as np

from SparseMat2 import SparseMat

__author__ = "Marco Bommarito"
__email__ = "bommaritom@carleton.edu"
"""
DESCRIPTION:

This file contains a single method: svd_decomp.

svd_decomp takes in a square SparseMat M and returns three SparseMat objects
corresponding to the SVD decomposition of the underlying matrix M:

M = U * SIGMA * V

You can then extract the data from the returned objects.

At the bottom of this page, code is proposed to convert a square matrix into
a SparseMat, which can be inputted to svd_decomp.

To use this method on a rectangular matrix, it should first be extended into a
square matrix, and then converted into the SparseMat data structure.
"""


def svd_decomp(M):

    #----SETUP----#
    #extract data from M object
    value = []
    col_index = []
    row_index = []
    for x in M.value:
        value.append(x)
    for j in M.col_index:
        col_index.append(j)
    for i in M.row_index:
        row_index.append(i)
    SIGMA = SparseMat(value, col_index, row_index)

    dim = len(SIGMA.row_index) - 1

    #create diagonal matrices

    U = SparseMat([1] * dim, list(range(dim)), list(range(dim + 1)))
    V = SparseMat([1] * dim, list(range(dim)), list(range(dim + 1)))
    #-------------#

    #QR FACTORIZATION
    for j in range(0, dim - 1):
        for i in reversed(range(j + 1, dim)):
            #print('' + str(i) + ', ' + str(j))
            if SIGMA.value_at(i, j) != 0:
                row = i
                col = j

                kill = SIGMA.value_at(row, col)
                top = SIGMA.value_at(col, col)

                hyp = np.sqrt(kill**2 + top**2)
                cos = top / hyp
                sin = -kill / hyp

                U.givens_post(col, row, cos, sin)
                SIGMA.givens_pre(col, row, cos, sin)

    #BIDIAGONALIZATION
    for i in range(0, dim - 2):
        for j in reversed(range(i + 2, dim)):

            if SIGMA.value_at(i, j) != 0:

                #post-multiply
                row = i
                col = j

                kill = SIGMA.value_at(row, col)
                top = SIGMA.value_at(row, col - 1)

                hyp = np.sqrt(top**2 + kill**2)
                cos_post = top / hyp
                sin_post = -kill / hyp

                SIGMA.givens_post(col - 1, col, cos_post, sin_post)
                V.givens_pre(col - 1, col, cos_post, sin_post)

                #pre-multiply
                kill = SIGMA.value_at(col, col - 1)
                top = SIGMA.value_at(col - 1, col - 1)
                hyp = np.sqrt(top**2 + kill**2)

                if hyp != 0:
                    cos_pre = top / hyp
                    sin_pre = -kill / hyp

                    SIGMA.givens_pre(col - 1, col, cos_pre, sin_pre)
                    U.givens_post(col - 1, col, cos_pre, sin_pre)

    #ITERATIVE PROCESS

    #threshold = how many nonzero elements we tolerate off the diagonal
    #accuracy  = how many decimal places we round to regard a value as equal to 0
    #Increase threshold, decrease accuracy = faster but lossier.
    THRESHOLD = 0
    ACCURACY = 5

    #-begin algorithm-
    amt_nonzero = dim
    while amt_nonzero > THRESHOLD:

        amt_nonzero = 0

        for i in range(0, dim - 1):

            if round(SIGMA.value_at(i, i + 1), ACCURACY) != 0:
                amt_nonzero += 1

            kill = SIGMA.value_at(i, i + 1)
            top = SIGMA.value_at(i, i)
            hyp = np.sqrt(top**2 + kill**2)

            if hyp != 0:
                cos_post = top / hyp
                sin_post = -kill / hyp

                SIGMA.givens_post(i, i + 1, cos_post, sin_post)
                V.givens_pre(i, i + 1, cos_post, sin_post)

            kill = SIGMA.value_at(i + 1, i)
            top = SIGMA.value_at(i, i)
            hyp = np.sqrt(top**2 + kill**2)

            if hyp != 0:
                cos_pre = top / hyp
                sin_pre = -kill / hyp

                SIGMA.givens_pre(i, i + 1, cos_pre, sin_pre)
                U.givens_post(i, i + 1, cos_pre, sin_pre)

        #print(amt_nonzero)

    output = []
    output.append(U)
    output.append(SIGMA)
    output.append(V)

    return output


# def to_sparse(M):

#     value = []
#     col_index = []
#     row_index = [0]

#     for i in range(0, len(M)):
#         for j in range(0, len(M[i])):
#             value.append(M[i][j])
#             col_index.append(j)
#         row_index.append(len(col_index))

#     return SparseMat(value, col_index, row_index)

# svd_decomp(to_sparse(M))
