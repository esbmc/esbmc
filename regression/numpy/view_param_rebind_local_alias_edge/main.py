import numpy as np

def rebind_alias(row):
    alias = row
    alias = None
    return row[0]

a = np.array([5, 6, 7])
v = a[0:]
result = rebind_alias(v)
assert result == 5
