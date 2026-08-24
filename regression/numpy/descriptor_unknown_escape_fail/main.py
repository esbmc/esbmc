import numpy as np


def consume(v):
    return v[0][0]


a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

assert consume(t) == 1
