import numpy as np

a = np.array([0, 2, -1], dtype=np.bool_)
b = np.zeros(2, dtype="bool")
assert not a[0]
assert a[1] and a[2]
assert not b[1]
