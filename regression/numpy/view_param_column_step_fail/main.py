import numpy as np

def set_last(col):
    n = len(col)
    col[n - 1] = 9
    return n

a = np.array([[1, 2], [3, 4], [5, 6]])
v = a[::2, 0]
n = set_last(v)
assert n == 2
assert a[2][0] == 9
assert a[1][0] == 9
