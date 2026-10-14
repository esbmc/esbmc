import numpy as np


def make():
    a = np.array([5, 1, 3])
    return a


result = make()
idx = np.searchsorted(result, 2, sorter=np.argsort(result))

assert idx == 0
