import numpy as np


def total(v):
    return np.sum(v)


a = np.array([[1, 2, 3], [4, 5, 6]])
col = a[:, 1]
assert total(col) == 8
