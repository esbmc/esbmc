import numpy as np


def make():
    return np.array([1, 2]) / 2


def first(a):
    return a[0]


assert first(make()) == 0.5
