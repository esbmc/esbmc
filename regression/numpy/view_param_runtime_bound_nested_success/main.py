import numpy as np

def inner(v):
    v[0] = 50
    return len(v)

def outer(row):
    sub = row[1:]
    k = inner(sub)
    return k + len(row)

a = np.array([1, 2, 3, 4])
n = 3
v = a[0:n]
x = outer(v)
assert x == 5
assert a[1] == 50
