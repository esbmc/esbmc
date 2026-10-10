import numpy as np

def modify_copy(row):
    c = row.copy()
    c[0] = 99
    return row[0]

a = np.array([10, 20, 30])
v = a[0:]
result = modify_copy(v)
assert result == 10
