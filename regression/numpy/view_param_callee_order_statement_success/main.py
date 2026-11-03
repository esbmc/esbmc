import numpy as np

def outer(row):
    inner(row)
    return row[2]

def inner(row):
    row[2] = 42

a = np.array([1, 2, 3, 4])
v = a[::1]
r = outer(v)
assert r == 42
assert a[2] == 42
