import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
s = a[::2]
assert sum(s) == 12
assert min(s) == 3
assert max(s) == 5
