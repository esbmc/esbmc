import numpy as np


def scale(a):
    return a * 2


x = np.array([1, 2, 3])
r = scale(x)
x[0] = 9
assert r[0] == 2
assert x.T[-1] == 3
