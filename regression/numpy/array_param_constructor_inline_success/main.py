import numpy as np


def corner(m):
    return m[1][1]


def total(a):
    s = 0.0
    for v in a:
        s += v
    return s


x = np.eye(2)
assert corner(x) == 1.0
assert corner(np.eye(2)) == 1.0
r = corner(np.identity(2))
assert r == 1.0
assert total(np.arange(1.0, 4.0)) == 6.0
