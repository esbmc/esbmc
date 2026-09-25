import numpy as np


def identity(a):
    return a


x = np.array([False, False, True, False])
y = identity(x)

assert y.any() == True
assert y.all() == False
