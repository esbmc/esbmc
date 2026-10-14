import numpy as np


def make():
    return np.array([1, 3, 5, 7])


i = np.searchsorted(make(), 4)

assert i == 2
