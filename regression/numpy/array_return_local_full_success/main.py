import numpy as np


def make():
    a = np.full((2, 2), 7)
    return a


y = make()

assert y[0, 0] == 7
assert y[0, 1] == 7
assert y[1, 0] == 7
assert y[1, 1] == 7
