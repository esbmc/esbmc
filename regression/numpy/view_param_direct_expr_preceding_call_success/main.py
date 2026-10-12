import numpy as np

def setk(row, k):
    row[0] = k
    return row[0]

n = 2

def grow():
    global n
    n = 3
    return 0

a = np.array([1, 2, 3, 4])
x = grow() + setk(a[0:n], 7)
assert x == 7
assert a[0] == 7
assert a[2] == 3
