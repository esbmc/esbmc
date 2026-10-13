import numpy as np

def empty_tail(row):
    sub = row[3:]
    return len(sub)

a = np.array([1, 2, 3])
v = a[0:]
assert empty_tail(v) == 0
assert a[2] == 3
