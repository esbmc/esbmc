import numpy as np

def passthrough(row):
    row[0] = 42
    return row

a = np.array([1, 2, 3])
v = a[0:]
x = passthrough(v)[0]
assert x == 42
