import numpy as np

def set_last(v):
    v[-1] = 7
    return len(v)

def outer(row):
    sub = row[1:]
    inner = sub[1:4]
    return set_last(inner)

a = np.array([1, 2, 3, 4, 5, 6])
v = a[0:]
assert outer(v) == 3
assert a[4] == 7
assert a[5] == 6
