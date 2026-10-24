import numpy as np

def modify_via_alias(row):
    alias = row
    alias[1] = 99
    return row[1]

a = np.array([1, 2, 3])
v = a[0:]
result = modify_via_alias(v)
assert result == 99
assert a[1] == 99
