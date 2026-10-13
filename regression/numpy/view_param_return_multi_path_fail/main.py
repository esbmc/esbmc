import numpy as np

def pick(row, other, flag):
    row[0] = 1
    if flag:
        return row
    return other

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
v = a[0:]
w = b[0:]
r = pick(v, w, True)
assert r[0] == 1
