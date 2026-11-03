import numpy as np

def make():
    return np.array([5, 6, 7, 8])

def set_first(row):
    row[0] = 9
    return len(row) + row[1]

a = make()
v = a[1:3]
n = set_first(v)
assert n == 9
assert a[1] == 9
