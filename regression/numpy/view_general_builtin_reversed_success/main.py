import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
r = list(reversed(v))
assert r[0] == 1
assert r[1] == 4
assert r[2] == 1
