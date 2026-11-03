import numpy as np

def swap_corner(mat):
    mat[0][1] = 7
    return mat[1][0]

a = np.array([[1, 2], [3, 4], [5, 6]])
v = a[1:]
assert swap_corner(v) == 5
assert a[1][1] == 4
