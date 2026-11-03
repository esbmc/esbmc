import numpy as np


def total(v):
    return np.sum(v)


def as_list(v):
    return v.tolist()


def tiled(a):
    return np.broadcast_to(a, (2, 3))


a = np.array([[1, 2, 3], [4, 5, 6]])
col = a[:, 1]
assert total(col) == 7
assert as_list(col) == [2, 5]

b = np.array([1, 2, 3])
t = tiled(b)
assert t[1][2] == 3
assert t.shape == (2, 3)
