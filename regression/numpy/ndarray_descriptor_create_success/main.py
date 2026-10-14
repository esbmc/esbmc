import numpy as np

a = np.array([[1, 2], [3, 4]])
assert a[0][0] == 1
assert a[0][1] == 2
assert a[1][0] == 3
assert a[1][1] == 4
assert a.ndim == 2
assert a.shape[0] == 2
assert a.shape[1] == 2
