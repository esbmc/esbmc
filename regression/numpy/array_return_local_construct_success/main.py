import numpy as np


def construct_return():
    a = np.zeros(3)
    a[0] = 5
    a[1] = 10
    a[2] = 15
    return a


result = construct_return()
assert result[0] == 5
assert result[1] == 10
assert result[2] == 15
