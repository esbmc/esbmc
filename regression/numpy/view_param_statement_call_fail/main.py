import numpy as np

def double_all(row):
    row[0] = row[0] * 2
    row[1] = row[1] * 2

a = np.array([3, 4, 5])
v = a[0:]
double_all(v)
assert a[0] == 3
