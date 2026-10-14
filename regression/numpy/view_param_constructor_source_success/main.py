import numpy as np

def set_first(row):
    row[0] = 9
    return len(row) + row[1]

def set_half(row):
    row[0] = 0.5
    return len(row)

z = np.zeros(4, dtype=int)
v = z[1:3]
n = set_first(v)
assert n == 2
assert z[1] == 9

o = np.ones(5, dtype=int)
w = o[2:4]
m = set_first(w)
assert m == 3
assert o[2] == 9

f = np.zeros(3)
u = f[1:]
k = set_half(u)
assert k == 2
assert f[1] == 0.5
