import numpy as np


def column(a):
    return a[:, 2]


a = np.array([[1, 2, 3], [4, 5, 6]])
c = column(a)
d = c
assert len(d) == 2
assert d.shape == (2,)
assert d.tolist() == [3, 6]
assert np.sum(d) == 9
e = np.copy(d)
a[0][2] = 0
assert d[0] == 0
assert e[0] == 3

box = {'c': c}
assert np.max(box['c']) == 6
