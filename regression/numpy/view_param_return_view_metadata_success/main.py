import numpy as np

def bump(row):
    row[0] = row[0] + 1
    return row

a = np.array([1, 2, 3, 4])
v = a[1:4]
n = len(bump(v))
assert n == 3
assert bump(v).shape == (3,)
assert bump(v).ndim == 1
assert bump(v).size == 3
assert a[1] == 6
