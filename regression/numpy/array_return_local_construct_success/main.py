import numpy as np


def make():
    a = np.zeros(3)
    a[0] = 5
    a[2] = 9
    return a


y = make()

assert y[0] == 5
assert y[1] == 0
assert y[2] == 9
