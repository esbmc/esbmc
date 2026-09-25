import numpy as np


def transposed(a):
    return np.transpose(a)


x = np.array([[1, 2], [3, 4]])
y = transposed(x)

assert y[0][1] == 3
assert y[1][0] == 2
