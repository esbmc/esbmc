import numpy as np


def one():
    return 1


def two():
    return 2


def pick(v):
    n = one()
    n = two()
    return v[n]


a = np.zeros((2, 3), dtype=int)
row = a[0]
assert pick(row) == 3
