import numpy as np

def touch(row):
    row[0] = 0
    return row[3]

a = np.array([1, 2, 3, 4])
v = a[0:4]
w = a[0:2]
r1 = touch(v)
r2 = touch(row=w)
