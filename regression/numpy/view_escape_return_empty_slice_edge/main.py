import numpy as np


def nothing(a):
    return a[2:2]


a = np.array([10, 20, 30, 40, 50])
e = nothing(a)
assert len(e) == 0
assert e.shape == (0,)
try:
    x = e[0]
    assert False
except IndexError:
    pass
