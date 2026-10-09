import numpy as np

def bump(col):
    old = col[1]
    col[1] = old + 10
    return len(col)

a = np.array([[1, 2], [3, 4], [5, 6]])
v = a[0:2, 1]
n = bump(v)
assert n == 2
assert a[1][1] == 14
assert a[2][1] == 6
