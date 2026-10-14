import numpy as np


def one():
    return 1


def pick(v):
    unused = 5
    n = one()
    return v[n]


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
assert pick(row) == 3
