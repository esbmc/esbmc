import numpy as np


def first_row(a):
    r = a[0]
    return r


a = np.array([[1, 2, 3], [4, 5, 6]])
v = first_row(a)
assert v[1] == 2
v[1] = 9
assert a[0][1] == 9
