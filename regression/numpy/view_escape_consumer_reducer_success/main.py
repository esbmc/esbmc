import numpy as np


def column(a):
    return a[:, 1]


a = np.array([[1, 2, 3], [4, 5, 6]])
c = column(a)
assert np.sum(c) == 7
assert np.min(c) == 2
assert np.max(c) == 5
assert np.mean(c) == 3.5

box = (a[0],)
assert np.sum(box[0]) == 6
assert box[0].any()
assert box[0].all()
