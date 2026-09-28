import numpy as np


def make():
    a = np.array([1, 3, 5, 7])
    return a


result = make()
i = np.searchsorted(result, 4)
assert i == 2
