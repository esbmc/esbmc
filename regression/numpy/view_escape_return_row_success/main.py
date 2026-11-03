import numpy as np


def first_row(a):
    return a[0]


a = np.array([[1, 2, 3], [4, 5, 6]])
r = first_row(a)
assert r[1] == 2
r[1] = 9
assert a[0][1] == 9
a[0][2] = 7
assert r[2] == 7
