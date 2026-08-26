import numpy as np


def pick(cond):
    a = np.array([1, 2, 3])
    c = np.array([10, 20, 30, 40, 50])
    if cond:
        b = np.ravel(a)
    else:
        b = np.ravel(c)
    return len(b)


assert pick(True) == 3
