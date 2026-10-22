import numpy as np


def last(a):
    return a[a.shape[0] - 1]


x = np.linspace(0, 1, 5)
assert last(x) == 0.75
