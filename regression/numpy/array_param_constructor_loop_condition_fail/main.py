import numpy as np


def first(a):
    return a[0]


n = 0
while first(np.zeros(2)) == 1.0:
    n += 1
assert n == 1
