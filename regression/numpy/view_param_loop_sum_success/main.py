import numpy as np

def total(row):
    s = 0
    for i in range(len(row)):
        s = s + row[i]
    return s

a = np.array([1, 2, 3, 4, 5])
v = a[1:4]
r = total(v)
assert r == 9
