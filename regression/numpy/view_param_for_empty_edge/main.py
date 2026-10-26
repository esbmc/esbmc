import numpy as np

def fill_empty(row):
    n = len(row)
    for i in range(n):
        row[i] = 99

a = np.array([1, 2, 3])
v = a[0:0]
fill_empty(v)
assert a[0] == 1
