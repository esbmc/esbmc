import numpy as np

def some_but_not_all(row):
    some = np.any(row)
    every = np.all(row)
    return some and not every

a = np.array([0, 1, 0])
v = a[0:]
assert not some_but_not_all(v)
