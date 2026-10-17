import numpy as np


def first(b):
    return b[0]


a = np.array([1, 2, 3])
b = np.broadcast_to(a, (2, 3))
r = first(b)
assert r[2] == 3
r[0] = 9
