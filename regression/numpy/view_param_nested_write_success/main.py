import numpy as np

def inner(row):
    row[0] = 99

def outer(row):
    inner(row)
    return row[0]

a = np.array([1, 2, 3])
v = a[0:]
result = outer(v)
assert result == 99
assert a[0] == 99
