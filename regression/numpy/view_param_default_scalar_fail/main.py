import numpy as np

def add_offset(row, offset=5):
    row[0] = row[0] + offset
    return row[0]

a = np.array([1, 2, 3])
v = a[0:]
result = add_offset(v)
assert result == 99
