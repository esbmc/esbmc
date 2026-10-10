import numpy as np

def write_both(row):
    row[0] = 11
    row[1] = 22
    return row[0] + row[1]

a = np.array([1, 2, 3])
v = a[0:]
result = write_both(v)
assert result == 33
assert a[0] == 11
assert a[1] == 22
