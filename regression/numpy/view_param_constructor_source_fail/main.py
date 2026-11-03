import numpy as np

def set_first(row):
    row[0] = 9
    return len(row) + row[1]

z = np.ones(4, dtype=int)
v = z[1:3]
n = set_first(v)
assert n == 2
