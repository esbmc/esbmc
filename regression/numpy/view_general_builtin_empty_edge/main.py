import numpy as np

a = np.array([3, 1, 4, 1, 5])
v = a[1:4]
e = a[2:2]
assert sum(e) == 0
assert not any(e)
assert all(e)
assert len(tuple(e)) == 0
