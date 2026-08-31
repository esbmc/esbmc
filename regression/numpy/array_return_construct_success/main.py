import numpy as np


def make_array():
    return np.array([1, 2, 3])


def make_zeros():
    return np.zeros(3)


a = make_array()
b = make_zeros()

assert a[0] == 1
assert a[2] == 3
assert b[0] == 0
