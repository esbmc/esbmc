import numpy as np


def rows(v):
    return len(v)


def cut(a):
    return a[:, 1:3]


a = np.array([[1, 2, 3], [4, 5, 6]])
v = a[:, 1:3]
assert rows(v) == 2

r = cut(a)
assert len(r) == 2
assert r.shape == (2, 2)
assert r.ndim == 2
assert r.size == 4

box = [a[0], a.T]
assert len(box[0]) == 3
assert box[1].shape == (3, 2)
assert box[1].ndim == 2
assert box[1].size == 6
