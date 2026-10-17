import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
s = sorted(v, key=lambda x: -x)
assert s[0] == 4
assert s[2] == 1
assert v[0] == 1
assert max(v, key=lambda x: -x) == 1
assert min(v, key=lambda x: -x) == 4
t = sorted(a, key=abs)
assert t[0] == 1
assert a[0] == 3
r = sorted(v, reverse=True)
assert r[0] == 4
