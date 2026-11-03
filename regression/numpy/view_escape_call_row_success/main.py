import numpy as np


def second(row):
    return row[1]


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
assert second(row) == 2
a[0][1] = 9
assert second(row) == 9
