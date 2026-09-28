import numpy as np


def identity(a):
    return a


x = np.array([1, 2, 3, 4])
y = identity(x)

assert y.sum() == 10
assert y.mean() == 2.5
assert y.min() == 1
assert y.max() == 4
