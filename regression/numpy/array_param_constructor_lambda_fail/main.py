import numpy as np


def first(a):
    return a[0]


h = lambda: first(np.zeros(2))
assert h() == 0.0
