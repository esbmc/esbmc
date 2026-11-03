import numpy as np


def halves():
    return np.array([1, 2]) / 2


def bumped():
    b = np.array([1, 2]) + 1
    return b


def pick(a, i):
    return a[i]


h = halves()
assert h[0] == 0.5
assert bumped()[1] == 3
assert pick(np.array([5, 10, 15]), 2) == 15
