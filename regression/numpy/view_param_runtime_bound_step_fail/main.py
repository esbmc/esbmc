import numpy as np

def total(row):
    s = 0
    for i in range(len(row)):
        s = s + row[i]
    return s

a = np.array([1, 2, 3, 4, 5])
n = 4
v = a[1:n]
w = a[0:n:2]
r1 = total(v)
r2 = total(w)
assert r2 == 9
