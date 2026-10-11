import numpy as np

def outer(row):
    row[1] = 0
    k = inner(row)
    return k

def inner(row):
    row[0] = 4
    return len(row)

a = np.array([1, 2, 3, 4])
n = 3
v = a[0:n]
x = outer(v)
assert x == 4
