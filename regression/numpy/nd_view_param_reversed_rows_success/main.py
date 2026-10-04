import numpy as np


def flip(m):
    r = m[::-1]
    r[0][0] = 7
    return r[1][1]


a = np.array([[1, 2], [3, 4], [5, 6]])
x = flip(a)
assert x == 4
assert a[2][0] == 7
