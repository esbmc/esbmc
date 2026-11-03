import numpy as np


def make():
    return np.array([1, 2, 3])


def length(a):
    return a.shape[0]


assert length(make()) == 2
