import numpy as np

a = np.array([1, 2, 0, 3])
v = a[0:2]
assert all(v)
z = a[1:3]
assert not all(z)
