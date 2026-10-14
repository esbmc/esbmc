import numpy as np


def make():
    a = np.zeros(3)
    a[0] = 5
    return a


y = make()

assert y[0] == 5
assert y[1] == 0
