import numpy as np

def passthrough(row):
    row[0] = 42
    return row

a = np.array([1, 2, 3, 4])
v = a[0:2]
y = a[1] + passthrough(v)[0]
assert y == 44
assert a[0] == 42
