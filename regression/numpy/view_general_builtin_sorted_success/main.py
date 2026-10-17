import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
s = sorted(v)
assert s == [1, 1, 4]
assert a[2] == 4
