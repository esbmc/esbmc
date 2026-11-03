import numpy as np

def zero_last(row, k):
    row[len(row) - 1] = k
    return len(row)

def two():
    return 2

a = np.array([1, 2, 3, 4])
n = 3
v = a[1:n]
x = zero_last(v, two())
