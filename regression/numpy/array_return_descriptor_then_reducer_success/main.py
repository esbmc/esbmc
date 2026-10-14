import numpy as np


def transposed(a):
    return np.transpose(a)


x = np.array([[1, 2], [3, 4]])
y = transposed(x)

assert y.sum() == 10
assert y.max() == 4
