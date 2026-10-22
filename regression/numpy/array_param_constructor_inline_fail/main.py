import numpy as np


def corner(m):
    return m[1][1]


x = np.eye(2)
assert corner(x) == 1.0
assert corner(np.eye(2)) == 1.0
