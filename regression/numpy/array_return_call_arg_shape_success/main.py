import numpy as np


def make():
    return np.array([1, 2, 3])


def length(a):
    return a.shape[0]


def count(a):
    return a.size


assert length(make()) == 3
x = np.array([7, 8, 9])
assert length(x) == 3
assert count(x) == 3
