import numpy as np

def set_first(row):
    row[0] = 9
    return len(row)

def set_half(row):
    row[0] = 0.5
    return len(row)

n = 5
r = np.arange(n)
w = r[2:4]
m = set_first(w)
assert m == 2
assert r[2] == 9

f = np.arange(0.0, 4.0)
u = f[1:3]
k = set_half(u)
assert k == 2
assert f[1] == 0.5
