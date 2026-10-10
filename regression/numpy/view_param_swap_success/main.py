import numpy as np

def swap(row):
    tmp = row[0]
    row[0] = row[1]
    row[1] = tmp

a = np.array([10, 20, 30])
v = a[0:]
swap(v)
assert a[0] == 20
assert a[1] == 10
