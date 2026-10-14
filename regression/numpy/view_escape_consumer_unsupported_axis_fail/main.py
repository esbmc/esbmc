import numpy as np


def block(a):
    return a[:, 0:2]


a = np.array([[1, 2, 3], [4, 5, 6]])
b = block(a)
s = np.sum(b, keepdims=True)
