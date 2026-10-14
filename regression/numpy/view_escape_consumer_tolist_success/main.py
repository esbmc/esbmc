import numpy as np


def column(a):
    return a[:, 1]


a = np.array([[1, 2, 3], [4, 5, 6]])
c = column(a)
assert c.tolist() == [2, 5]

box = [a[1]]
assert box[0].tolist() == [4, 5, 6]
