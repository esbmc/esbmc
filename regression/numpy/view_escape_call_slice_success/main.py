import numpy as np


def ends(v):
    return v[0] + v[-1]


a = np.array([10, 20, 30, 40, 50])
v = a[1:4]
assert ends(v) == 60
