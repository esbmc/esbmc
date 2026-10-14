import numpy as np

def describe(row):
    sub = row[1:4]
    assert sub.size == 3
    return len(sub)

a = np.array([1, 2, 3, 4, 5])
v = a[0:]
assert describe(v) == 4
