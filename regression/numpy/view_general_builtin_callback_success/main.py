import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
s = sorted(v, key=lambda x: -x)
assert s[0] == 4
assert a[1] == 1
