import numpy as np

def inner(row):
    row[0] = 99
    return row[0]

def outer(row):
    x = inner(row)
    return x

a = np.array([1, 2, 3])
v = a[0:]
result = outer(v)
assert result == 99
assert a[0] == 99
