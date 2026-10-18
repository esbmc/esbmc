import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
assert max(v) == 5
