import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert len(v) == 3
assert len(a[::2]) == 3
