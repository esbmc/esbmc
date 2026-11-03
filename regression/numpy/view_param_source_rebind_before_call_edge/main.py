import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

a = np.array([1, 2, 3, 4])
old = a[0:2]
a = np.array([5, 6, 7, 8])
v = a[1:4]
n = set_first(v)
assert n == 3
assert a[1] == 9
assert old[0] == 1
