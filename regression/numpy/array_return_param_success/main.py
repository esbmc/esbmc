import numpy as np


def identity(a):
    return a


x = np.array([1, 2, 3])
y = identity(x)

assert y[0] == 1
assert y[2] == 3
