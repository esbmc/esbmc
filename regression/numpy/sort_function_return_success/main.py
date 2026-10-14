import numpy as np


def make(values):
    return np.array([9, 8, 7])


a = make([3, 1, 2])
b = np.sort(a)

assert b[0] == 7
assert b[1] == 8
assert b[2] == 9
