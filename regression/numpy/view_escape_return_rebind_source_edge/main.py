import numpy as np


def first_row(a):
    return a[0]


# Rebinding the source name leaves the returned view on the old storage.
a = np.array([[1, 2, 3], [4, 5, 6]])
r = first_row(a)
a = np.array([[7, 7, 7], [7, 7, 7]])
assert r[1] == 2
assert a[0][1] == 7
