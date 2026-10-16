import numpy as np

def write_every_other(row):
    sub = row[1::2]
    sub[1] = 8
    return sub[0] + len(sub)

a = np.array([1, 2, 3, 4, 5, 6])
v = a[0:]
assert write_every_other(v) == 5
assert a[3] == 8
assert a[2] == 3
