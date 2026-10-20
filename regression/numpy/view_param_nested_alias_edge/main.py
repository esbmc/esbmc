import numpy as np

def inner(row):
    alias = row
    alias[0] = 77

def outer(row):
    inner(row)
    return row[0]

a = np.array([1, 2, 3])
v = a[0:]
result = outer(v)
assert result == 77
