import numpy as np

def accumulate(row):
    total = sum(row)
    return total

a = np.array([1, 2, 3])
v = a[0:]
result = accumulate(v)
assert result == 6
