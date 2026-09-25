import numpy as np


def first_row(a):
    return a[0]


x = np.array([[1, 2], [3, 4]])
y = first_row(x)

assert y[0] == 1
assert y[1] == 2
