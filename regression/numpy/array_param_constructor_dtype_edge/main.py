import numpy as np


def first(a):
    return a[0]


def corner(m):
    return m[1][1]


assert first(np.array([1, 2], dtype=float)) == 1.0
e = np.eye(2)
assert corner(e) == 1.0
i = np.identity(2)
assert corner(i) == 1.0
small = np.ones(2, dtype=np.int8)
assert small[0] == 1
