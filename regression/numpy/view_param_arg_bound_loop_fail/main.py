import numpy as np

def fill_n(row, n, val):
    for i in range(n):
        row[i] = val

a = np.array([0, 0, 0, 0])
v = a[0:]
fill_n(v, 4, 99)
assert a[0] == 99
assert a[3] == 0
