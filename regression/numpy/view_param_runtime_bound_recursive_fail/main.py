import numpy as np

def rec(row):
    if len(row) <= 1:
        return row[0]
    rest = row[1:]
    r = rec(rest)
    return r + row[0] + len(row)

a = np.array([1, 2, 3])
n = 3
v = a[0:n]
x = rec(v)
assert x == 8
