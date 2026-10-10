import numpy as np

def setk(row, k):
    row[0] = k
    return row[0]

a = np.array([1, 2, 3, 4, 5, 6])
setk(a[2:4], 9)
assert a[2] == 3
