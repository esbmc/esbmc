import numpy as np


def identity(a):
    return a


x = np.array([1, 2, 3])
y = identity(x)
x[0] = 99
y = np.array([7, 8, 9])

assert x[0] == 99
assert y[0] == 7
assert y[1] == 8
assert y[2] == 9
