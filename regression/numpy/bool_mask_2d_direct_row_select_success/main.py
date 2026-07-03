import numpy as np

a = np.array([[1, 2], [3, 4]])
mask = np.array([True, False])

assert a[mask][0][0] == 1
assert a[mask][0][1] == 2
