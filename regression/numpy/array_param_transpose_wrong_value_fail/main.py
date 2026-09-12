import numpy as np


def transpose_param(a):
    b = np.transpose(a)
    return b[0, 1]


a = np.array([[1, 2], [3, 4]])
result = transpose_param(a)

# Wrong assertion to prove test detects incorrect values
assert result == 999
