import numpy as np


def array_return():
    a = np.array([1, 2, 3, 4])
    return a


result = array_return()
assert result[0] == 1
assert result[3] == 4
