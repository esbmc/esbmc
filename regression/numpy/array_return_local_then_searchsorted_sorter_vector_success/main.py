import numpy as np


def make():
    a = np.array([5, 1, 3])
    return a


result = make()
idx = np.searchsorted(result, [0, 2, 4, 6], sorter=np.argsort(result))

assert idx[0] == 0
assert idx[1] == 1
assert idx[2] == 2
assert idx[3] == 3
