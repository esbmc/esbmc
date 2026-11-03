import numpy as np


def make():
    return np.array([1, 2], dtype=float)


def make_single():
    return np.array([1, 2], dtype=np.float32)


def shift(a):
    return a[1] + 0.5


def double(a):
    return a * 2


assert shift(make()) == 2.5
r = double(make_single())
assert r[1] == 4.0
