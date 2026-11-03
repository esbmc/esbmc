import numpy as np

def fill(row, i):
    if i < 0:
        return 0
    row[i] = 7
    r = fill(row, i - 1)
    return r

a = np.array([1, 2, 3])
v = a[0:]
z = fill(v, 2)
assert a[0] == 7
assert a[2] == 7
