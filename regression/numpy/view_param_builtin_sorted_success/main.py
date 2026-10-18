import numpy as np

a = np.array([3, 1, 2])
v = a[0:]
s = sorted(v)
assert s[0] == 1
