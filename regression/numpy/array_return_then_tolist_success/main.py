import numpy as np


def identity(a):
    return a


x = np.array([1, 2, 3, 4])
y = identity(x)
lst = y.tolist()

assert lst[0] == 1
assert lst[3] == 4
