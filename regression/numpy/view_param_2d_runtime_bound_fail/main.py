import numpy as np

def corner(mat):
    r = len(mat)
    mat[r - 1][0] = 9
    return mat[0][1] + r

a = np.array([[1, 2], [3, 4], [5, 6]])
n = 2
v = a[0:n]
x = corner(v)
assert a[2][0] == 9
