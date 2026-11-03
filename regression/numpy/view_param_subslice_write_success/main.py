import numpy as np

def subslice_write(row):
    sub = row[1:]
    sub[0] = 77
    return row[1]

a = np.array([10, 20, 30])
v = a[0:]
result = subslice_write(v)
assert result == 77
assert a[1] == 77
