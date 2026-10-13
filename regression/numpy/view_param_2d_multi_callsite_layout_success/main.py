import numpy as np

def corner(mat):
    r = len(mat)
    mat[r - 1][0] = 9
    return mat[0][1] + r

a = np.array([[1, 2], [3, 4], [5, 6]])
v = a[0:2]
w = a[1:]
x = corner(v)
y = corner(w)
assert x == 4
assert y == 6
assert a[1][0] == 9
assert a[2][0] == 9
