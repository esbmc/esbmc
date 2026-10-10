import numpy as np

def read_elem(mat, i, j):
    return mat[i][j]

a = np.array([[1, 2, 3], [4, 5, 6]])
r = a[0]
val = read_elem(a, 1, 2)
assert val == 6
