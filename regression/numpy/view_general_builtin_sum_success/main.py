import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert sum(v) == 6
assert sum(v, 10) == 16
w = v
assert sum(w) == 6
