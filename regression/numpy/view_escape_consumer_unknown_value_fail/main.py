import numpy as np

calls = 0


def register(v):
    global calls
    calls = calls + 1


a = np.zeros((2, 3), dtype=int)
row = a[0]
register(row)
assert row.shape == (3,)
total = np.sum(row)
