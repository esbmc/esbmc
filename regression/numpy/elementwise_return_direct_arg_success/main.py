import numpy as np


def make():
    a = np.array([2, 4])
    return a / 2


def inline():
    return np.array([1, 2]) / 2


def first(x):
    return x[0]


assert first(make()) == 1.0
assert first(x=make()) == 1.0
assert first(inline()) == 0.5
