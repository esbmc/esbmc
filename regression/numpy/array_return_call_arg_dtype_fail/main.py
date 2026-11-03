import numpy as np


def make():
    return np.array([1, 2], dtype=float)


def shift(a):
    return a[1] + 0.5


assert shift(make()) == 3.0
