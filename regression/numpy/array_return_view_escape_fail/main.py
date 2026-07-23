import numpy as np


def first_row(a):
    return a[0]


x = np.array([[1, 2], [3, 4]])
y = first_row(x)
x[0][0] = 9

assert y[0] == 1
