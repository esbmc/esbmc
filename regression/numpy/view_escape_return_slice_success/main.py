import numpy as np


def middle(a):
    return a[1:3]


a = np.array([10, 20, 30, 40, 50])
m = middle(a)
assert len(m) == 2
assert m[-1] == 30
a[1] = 7
assert m[0] == 7
