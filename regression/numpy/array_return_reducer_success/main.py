import numpy as np


def identity(a):
    return a


x = np.array([1, 2, 3, 4])

assert np.sum(identity(x)) == 10
assert np.mean(identity(x)) == 2.5
assert np.min(identity(x)) == 1
assert np.max(identity(x)) == 4
