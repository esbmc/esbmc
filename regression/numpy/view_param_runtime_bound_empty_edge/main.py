import numpy as np

def count_twice(row):
    n = len(row)
    return n + n

a = np.array([1, 2, 3, 4])
n = 2
v = a[n:n]
r = count_twice(v)
assert r == 0
assert a[2] == 3
