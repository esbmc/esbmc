import numpy as np

g = 0


def first_row(a):
    global g
    g = 1
    return a[0]


x = np.array([[1, 2], [3, 4]])
first_row(x)

assert g == 0
