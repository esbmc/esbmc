import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
l = list(v)
v[0] = 9
assert l[0] == 9
