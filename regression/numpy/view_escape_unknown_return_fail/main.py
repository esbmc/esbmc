import numpy as np


def relay(v):
    return external_filter(v)


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
r = relay(row)
x = r[0]
