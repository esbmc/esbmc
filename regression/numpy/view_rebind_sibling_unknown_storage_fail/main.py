import numpy as np


def first_two(a):
    x = a[1:3]
    y = a[1:3]
    a = np.array([9, 9, 9, 9])
    x[0] = 7
    assert y[0] == 7


first_two(np.array([1, 2, 3, 4]))
