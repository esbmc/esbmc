import numpy as np

def corner(mat):
    r = len(mat)
    mat[r - 1][0] = 9
    return mat[0][1] + r

a = np.array([[1, 2], [3, 4], [5, 6]])
v = a[0:2]
w = a[0:]
x = corner(v)
y = corner(w)
assert y == 4
