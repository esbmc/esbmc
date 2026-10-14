import numpy as np


def make():
    a = np.array([3, 1])
    return a


idx = np.searchsorted(make(), 2)

assert idx == 1
