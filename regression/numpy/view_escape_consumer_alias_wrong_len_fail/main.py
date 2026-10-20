import numpy as np


def column(a):
    return a[:, 2]


a = np.array([[1, 2, 3], [4, 5, 6]])
c = column(a)
d = c
assert len(d) == 3
