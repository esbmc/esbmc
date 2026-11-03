import numpy as np


def double(a):
    return a * 2


x = np.array([1.0, 2.0])
r = double(x)
assert r[2] == 0.0
