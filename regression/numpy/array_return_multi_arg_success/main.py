import numpy as np


def pick_first(a, b):
    return a


x = np.array([1, 2, 3])
y = pick_first(x, 5)

assert y[0] == 1
