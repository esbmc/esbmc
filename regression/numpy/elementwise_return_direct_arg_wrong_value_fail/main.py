import numpy as np


def make():
    a = np.array([2, 4])
    return a / 2


def first(x):
    return x[0]


assert first(make()) == 2.0
