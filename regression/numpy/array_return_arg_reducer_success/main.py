import numpy as np


def identity(a):
    return a


x = np.array([3, 1, 4])

assert np.argmin(identity(x)) == 1
assert np.argmax(identity(x)) == 2
