import numpy as np

def f(row):
    row[0] = 7
    return row[0]

a = np.array([1, 2, 3, 4])
v = a[1:3]
w = v
assert f(w) == 7
assert a[1] == 7
