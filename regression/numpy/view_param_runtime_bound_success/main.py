import numpy as np

def first_two(row):
    n = len(row)
    return row[0] + row[n - 1]

a = np.array([1, 2, 3, 4])
n = 3
v = a[0:n]
assert first_two(v) == 4
