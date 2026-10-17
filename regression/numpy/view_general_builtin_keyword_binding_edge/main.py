import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert sum(v, start=10) == 16
assert max(v, default=0) == 4
assert min(a[2:2], default=7) == 7
