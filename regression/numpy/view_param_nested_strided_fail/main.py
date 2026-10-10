import numpy as np

def inner(row):
    row[1] = 50

def outer(row):
    inner(row)
    return row[1]

a = np.array([1, 2, 3, 4, 5, 6])
v = a[::2]
r = outer(v)
assert r == 50
assert a[2] == 50
assert a[1] == 50
