import numpy as np

def touch(row, k):
    row[0] = k
    return row[1]

a = np.array([1, 2, 3, 4])
v = a[0:2]
w = a[2:4]
assert touch(v, 7) == 2
assert touch(k=8, row=w) == 4
assert a[0] == 7
assert a[2] == 3
