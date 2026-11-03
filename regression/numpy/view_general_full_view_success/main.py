import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

f = np.full(5, 7)
w = f[1:4]
n = set_first(w)
assert n == 3
assert f[1] == 9
assert f[0] == 7
assert sum(w) == 23
