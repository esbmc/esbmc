import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
w = v[1:]
assert sum(w) == 5
assert min(w) == 1
w[0] = 9
assert sum(v) == 11
