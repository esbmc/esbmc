import numpy as np

def swap_corner(mat):
    x = mat[1][0]
    mat[0][1] = 7
    return x + mat[0][1]

a = np.array([[1, 2], [3, 4], [5, 6]])
v = a[1:]
assert swap_corner(v) == 12
assert a[1][1] == 7
assert a[0][1] == 2
