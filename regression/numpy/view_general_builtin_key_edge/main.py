import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
a[2] = 0
w = a[::2]
s = sorted(w, key=lambda x: 0)
assert s[0] == 3
assert s[1] == 0
assert s[2] == 5
assert max(a[1:4], key=lambda x: -x) == 0
