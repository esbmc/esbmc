import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
l = list(v)
assert l == [1, 4, 1]
r = list(a[::-2])
assert r == [5, 4, 3]
v[0] = 9
assert l[0] == 1
