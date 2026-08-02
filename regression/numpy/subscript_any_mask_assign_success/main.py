import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])
mask = np.array([True, False, True])
sel = a[mask]
assert sel[0][0] == 1
assert sel[0][1] == 2
assert sel[1][0] == 5
assert sel[1][1] == 6
