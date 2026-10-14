import numpy as np

def zero_last(row, k):
    row[len(row) - 1] = k
    return len(row)

a = np.array([1, 2, 3, 4])
n = 3
v = a[1:n]
c = True
x = c and zero_last(v, 0) > 0
