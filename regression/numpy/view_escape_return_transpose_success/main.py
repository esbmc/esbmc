import numpy as np


def flipped(a):
    return a.T


a = np.array([[1, 2, 3], [4, 5, 6]])
t = flipped(a)
assert t[2][0] == 3
a[0][2] = 7
assert t[2][0] == 7
