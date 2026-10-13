import numpy as np

def setk(row, k):
    row[0] = k
    return row[0]

a = np.array([1, 2, 3, 4])
v = a[0:2]
x = a[0] + setk(v, 7)
assert x == 8
y = a[0] + setk(a[0:2], 9)
assert y == 16
