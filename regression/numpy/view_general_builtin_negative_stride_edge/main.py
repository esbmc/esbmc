import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
r = a[::-2]
assert sum(r) == 12
assert max(r) == 5
t = tuple(r)
assert t[0] == 5
assert t[2] == 3
