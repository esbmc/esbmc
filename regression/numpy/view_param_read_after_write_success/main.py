import numpy as np

def rw(row):
    row[0] = 10
    row[1] = row[0] + 5
    return row[1]

a = np.array([1, 2, 3])
v = a[0:]
result = rw(v)
assert result == 15
assert a[1] == 15
