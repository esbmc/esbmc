import numpy as np


def cell(t):
    return t[2][0]


a = np.array([[1, 2, 3], [4, 5, 6]])
t = a.T
assert cell(t) == 3
