import numpy as np


def make():
    return np.array([3, 1, 2])


b = np.sort(make())
idx = np.argsort(make())

assert b[0] == 1
assert b[1] == 2
assert b[2] == 3
assert idx[0] == 1
assert idx[1] == 2
assert idx[2] == 0
