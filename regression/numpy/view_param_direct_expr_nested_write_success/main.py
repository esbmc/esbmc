import numpy as np

def setk(row, k):
    row[0] = k
    return row[0]

a = np.array([1, 2, 3, 4, 5, 6])
x = setk(a[0:2], 7) + 1
assert a[0] == 7
