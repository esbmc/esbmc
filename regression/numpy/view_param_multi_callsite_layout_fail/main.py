import numpy as np

def set_last(row):
    k = len(row)
    row[k - 1] = 0
    return k

a = np.array([1, 2, 3, 4, 5, 6])
v = a[0:2]
w = a[2:6]
n1 = set_last(v)
n2 = set_last(w)
assert n2 == 2
