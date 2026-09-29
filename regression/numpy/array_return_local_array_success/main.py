import numpy as np


def make():
    a = np.array([1, 2, 3])
    return a


y = make()

assert y[0] == 1
assert y[1] == 2
assert y[2] == 3
