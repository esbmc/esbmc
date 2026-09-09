import numpy as np


def branch_return(x):
    if x > 0:
        return np.array([1, 2, 3])
    else:
        return np.array([4, 5, 6])


r = branch_return(5)
assert r[0] == 1

s = branch_return(-1)
assert s[0] == 4
