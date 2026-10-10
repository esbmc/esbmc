import numpy as np


def one():
    return 1


def two():
    return 2


# Inlining `n` would run one() after two().
def pick(v):
    n = one()
    return two() + v[n]


a = np.zeros((2, 3), dtype=int)
row = a[0]
assert pick(row) == 4
