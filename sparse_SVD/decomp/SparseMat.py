import numpy as np

__author__ = "Marco Bommarito"
__email__  = "bommaritom@carleton.edu"

class SparseMat:

    """
    DESCRIPTION:

    SparseMat is an implementation of the sparse matrix data structure, with SVD computation in mind.
    This class is dedicated to the algorithm svd_decomp, and should not be used for any other purpose.
    If you're looking for help using the svd method, skip over this file.
    """

    """
    METHODS OVERVIEW:

    __init__(self, value, col_index, row_index):    Stores the input as instance variables, also stores size.

                                                    Very important note: There is no input sanitation but the input should
                                                    be a SQUARE MATRIX IN CSR/CRS/Yale FORMAT:
                                                    https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

                                                    Note: Why a square matrix? Helps with some of the logic.
                                                    At worst this may accrue a minor, "linear" (<= n) increase 
                                                    in space complexity when the matrix has more columns than rows.
                                                    (If more rows than columns we don't get this complexity increase, surprisingly.)


    __str__(self):                                  Overrides the default print method, self-explanatory.
                                                    For debugging.


    value_at(self, i, j):                           Returns the value at (i,j). "Linear" (<= n)
                                                    time complexity because of data compression.


    replace(self, i, j, v):                         Replaces the value at (i,j) with v. Lot of cases here 
                                                    depending on whether v = 0 or the value at (i,j) is 0.
                                                    2 return statements is somewhat confusing but it works.
                                                    Can probably clean it up.


    givens_pre(self, i, k, c, s):                   A counterclockwise rotation of theta radians through 
                                                    the i-k plane. 
                                                    Here i and k are rows, c = cos(theta), s = sin(theta).
                                                    Time complexity on the lower end of O(n^2).


    givens_post(self, i, k, c, s):                  The post-multiplication analog of the above. 
                                                    Here i and k are columns.

    """


    def __init__(self, value, col_index, row_index):

        self.DECIMALS = 8
        #can change this depending on how accurately we want to store the data.

        self.value     = value
        self.col_index = col_index
        self.row_index = row_index

        self.dim = len(self.row_index)-1


    def __str__(self):
        
        ACCURACY = 3
        #this does not affect any data, just how it's printed

        output = ''
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                entry = str(round(self.value_at(i, j), ACCURACY))
                output += entry
                for char in range(0, ACCURACY + 4 - len(entry)):
                    output += ' '
            output += '\n'

        return output


    def value_at(self, i, j):

        for col in range(self.row_index[i], self.row_index[i+1]):
            if self.col_index[col] == j:
                return self.value[col]
        return 0

    def replace(self, i, j, v):

        for col in range(self.row_index[i], self.row_index[i+1]):
            if self.col_index[col] == j:
                if v == 0:
                    self.value.pop(col)
                    self.col_index.pop(col)
                    for d in range(i+1,len(self.row_index)):
                        self.row_index[d] -= 1
                else:
                    self.value[col] = v
                return

        if v != 0:

            #c is the index of the desired entry of col_index.
            c = self.row_index[i]
            for col in range(self.row_index[i], self.row_index[i+1]):
                #if self.row_index[c] > self.row_index[col] and self.row_index[c] < j:
                if self.col_index[col] > self.col_index[c] and self.col_index[col] < j:
                    c = col

            self.value.insert(c, v)
            self.col_index.insert(c, j)
            for d in range(i+1,len(self.row_index)):
                self.row_index[d] += 1
            return

    def givens_pre(self, i, k, c, s):
        #iterate through COLS, assume square matrix.
        for j in range(0, len(self.row_index)-1):
            t1 = self.value_at(i, j)
            t2 = self.value_at(k, j)
            self.replace(i, j, round((c * t1) - (s * t2), self.DECIMALS))
            self.replace(k, j, round((s * t1) + (c * t2), self.DECIMALS))

    def givens_post(self, i, k, c, s):
        #iterate through ROWS
        for j in range(0, len(self.row_index)-1):
            t1 = self.value_at(j, i)
            t2 = self.value_at(j, k)
            self.replace(j, i, round((c * t1) - (s * t2), self.DECIMALS))
            self.replace(j, k, round((s * t1) + (c * t2), self.DECIMALS))

