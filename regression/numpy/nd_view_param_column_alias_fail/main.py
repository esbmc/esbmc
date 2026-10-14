import numpy as np


def col(m):
    c = m[:, 1]
    c[0] = 7
    return c[1]


a = np.array([[1, 2], [3, 4]])
x = col(a)
assert x == 3
assert a[0][1] == 7
