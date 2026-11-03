import numpy as np


def first(b):
    return b[0]


a = np.array([1, 2, 3])
b = np.broadcast_to(a, (2, 3))
r = first(b)
c = np.copy(r)
c[0] = 9
assert c[0] == 9
assert a[0] == 1
assert r[0] == 1
