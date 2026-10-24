import numpy as np

def set_last(row):
    k = len(row)
    row[k - 1] = 0
    return k

a = np.array([1, 2, 3, 4, 5, 6])
v = a[0:2]
w = a[2:6]
s = a[::2]
n1 = set_last(v)
n2 = set_last(w)
n3 = set_last(s)
assert n1 == 2
assert n2 == 4
assert n3 == 3
assert a[1] == 0
assert a[5] == 0
assert a[4] == 0
assert a[0] == 1
