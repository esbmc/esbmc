import numpy as np

calls = 0


def register(v):
    global calls
    calls = calls + 1


def same(v):
    return v


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
register(row)
r = same(row)
