import numpy as np


def make(values):
    return np.array([9, 8, 7])


a = make([3, 1, 2])
b = np.sort(a)

assert b[0] == 1
