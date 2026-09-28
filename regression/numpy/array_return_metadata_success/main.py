import numpy as np


def identity(a):
    return a


x = np.array([1, 2, 3])
y = identity(x)

assert len(y) == 3
assert y.shape[0] == 3
assert y.ndim == 1
