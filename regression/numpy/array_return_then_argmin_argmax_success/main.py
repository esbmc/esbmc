import numpy as np


def make():
    return np.array([3, 1, 4, 1, 5])


y = make()

assert y.argmin() == 1
assert y.argmax() == 4
