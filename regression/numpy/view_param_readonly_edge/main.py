import numpy as np

def read_only_access(row):
    return row[0] + row[1]

a = np.array([5, 10, 15])
v = a[0:]
result = read_only_access(v)
assert result == 15
