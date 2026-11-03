import numpy as np

def outer(row):
    k = inner(row)
    return k + row[2]

def inner(row):
    row[2] = 42
    return len(row)

a = np.array([1, 2, 3, 4])
v = a[0:]
r = outer(v)
assert a[2] == 3
