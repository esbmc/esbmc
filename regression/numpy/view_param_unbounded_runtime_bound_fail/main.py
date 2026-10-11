import numpy as np

def zero_last(row):
    row[len(row) - 1] = 0
    return len(row)

a = np.array([1, 2, 3, 4])
n = 3
v = a[1:n]
x = zero_last(v) + 1
assert x == 3
