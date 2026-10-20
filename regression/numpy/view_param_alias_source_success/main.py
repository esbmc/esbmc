import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

a = np.array([1, 2, 3, 4])
b = a
v = b[1:3]
n = set_first(v)
assert n == 2
assert a[1] == 9
