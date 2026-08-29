import numpy as np


def read_local(a):
    row = a[0]
    assert row[0] == 1
    assert row[1] == 2


x = np.array([[1, 2], [3, 4]])
read_local(x)
assert x[0][0] == 1
