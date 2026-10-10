import numpy as np

def f(mat):
    x = mat[1][0]
    mat[0][1] = 7
    n = len(mat)
    return x + mat[0][1] + n

a = np.array([[1, 2], [3, 4], [5, 6]])
v = a[0:]
assert f(v) == 13
assert a[0][1] == 7
