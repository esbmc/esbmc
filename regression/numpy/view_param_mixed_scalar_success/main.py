import numpy as np

def add_offset(row, offset):
    row[0] = row[0] + offset
    return row[0]

a = np.array([1, 2, 3])
v = a[0:]
result = add_offset(v, 10)
assert result == 11
assert a[0] == 11
