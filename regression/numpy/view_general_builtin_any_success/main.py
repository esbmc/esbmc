import numpy as np

a = np.array([0, 0, 2, 0])
v = a[1:3]
assert any(v)
z = a[0:2]
assert not any(z)
