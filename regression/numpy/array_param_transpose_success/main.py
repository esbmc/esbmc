import numpy as np


def transpose_param(a):
    # Test transpose within function on parameter
    b = np.transpose(a)
    c = a.T
    d = a.transpose()
    # All should have swapped shape
    return b[0, 1], c[1, 0], d[0, 1]


a = np.array([[1, 2], [3, 4]])
v1, v2, v3 = transpose_param(a)

# a is [[1, 2], [3, 4]], so a.T is [[1, 3], [2, 4]]
# b[0, 1] = 3, c[1, 0] = 2, d[0, 1] = 3
assert v1 == 3
assert v2 == 2
assert v3 == 3
