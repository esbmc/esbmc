import numpy as np


def make():
    return np.array([3, 1, 2])


y = make()
idx = y.argsort()

assert idx[0] == 1
assert idx[1] == 2
assert idx[2] == 0
