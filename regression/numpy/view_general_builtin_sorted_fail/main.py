import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
s = sorted(v)
assert s[2] == 1
