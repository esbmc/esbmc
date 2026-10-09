import numpy as np

def find_first_zero(row):
    i = 0
    while row[i] != 0:
        i += 1
    return i

a = np.array([1, 2, 0, 4])
v = a[0:]
result = find_first_zero(v)
assert result == 2
