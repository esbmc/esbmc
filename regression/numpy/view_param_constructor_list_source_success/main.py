import numpy as np
def set_half(row):
    row[0] = 0.5
    return len(row)
def set_first(row):
    row[0] = 9
    return len(row)
r = np.linspace(0.0, 1.0, 5)
w = r[1:3]
m = set_half(w)
assert m == 2
assert r[1] == 0.5
assert r[2] == 0.5
e = np.eye(3)
row = e[1]
k = set_first(row)
assert k == 3
assert e[1][0] == 9
assert e[1][1] == 1
i = np.identity(2)
col = i[:, 1]
j = set_first(col)
assert j == 2
assert i[0][1] == 9
