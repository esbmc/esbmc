import numpy as np


def make():
    a = np.array([1, 3, 5, 7])
    return a


result = make()
idx = np.searchsorted(result, 4)

assert idx == 1
