import numpy as np

def setk(row, k):
    row[1] = k
    return row[1]

def double(x):
    y = x + x
    return y

a = np.array([1, 2, 3, 4])
r = double(setk(a[1:3], 5))
assert r == 10
assert a[2] == 5
