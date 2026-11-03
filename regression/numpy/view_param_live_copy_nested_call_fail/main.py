import numpy as np

def inner(row):
    row[1] = 5

def outer(row):
    sub = row[1:]
    inner(row)
    return sub[0]

a = np.array([10, 20, 30])
v = a[0:]
assert outer(v) == 20
