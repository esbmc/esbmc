import numpy as np

a = np.random.normal(size=(2, 3))
row = a[1]
col = a[:, 0]
b = np.random.random(4)
v = b[1:3]
v[0] = 7.0

assert row[0] == a[1][0]
assert col[1] == a[1][0]
assert b[1] == 7.0
