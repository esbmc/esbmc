import numpy as np

calls = 0


def register(v):
    global calls
    calls = calls + 1


def same(v):
    return v


a = np.zeros((2, 3), dtype=int)
row = a[0]
register(row)
r = same(row)
