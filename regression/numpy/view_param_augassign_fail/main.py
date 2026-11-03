import numpy as np

def scale(row, k):
    row[0] += k
    return row[0]

a = np.array([5, 6, 7])
v = a[0:]
result = scale(v, 3)
assert result == 5
