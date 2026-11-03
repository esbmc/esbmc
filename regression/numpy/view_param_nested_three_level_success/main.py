import numpy as np

def c(row):
    row[1] = 5
    return row[1]

def b(row):
    x = c(row)
    return x

def a_(row):
    y = b(row)
    return y + row[0]

m = np.array([[1, 2], [3, 4]])
v = m[1]
r = a_(v)
assert r == 8
assert m[1][1] == 5
