import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
t = tuple(v)
assert len(t) == 3
assert t[0] == 1
assert t[2] == 1
