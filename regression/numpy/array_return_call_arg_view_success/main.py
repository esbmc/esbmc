import numpy as np


def make():
    return np.array([[1, 2], [3, 4]])


def corner(m):
    return m.T[0][1]


assert corner(make()) == 3

x = np.array([[1, 2, 3], [4, 5, 6]])
assert x.T[0][1] == 4
assert x.T[2, 1] == 6
