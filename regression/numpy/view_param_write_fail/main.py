import numpy as np

def bump(row):
    old = row[1]
    row[1] = old + 5

a = np.array([1, 2, 3])
r = a[0:]
bump(r)
assert a[1] == 2
