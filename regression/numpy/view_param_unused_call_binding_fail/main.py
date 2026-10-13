import numpy as np

def sorted_first(row):
    ordered = sorted(row)
    return ordered[0]

a = np.array([3, 1, 2])
v = a[0:]
result = sorted_first(v)
assert result == 1
