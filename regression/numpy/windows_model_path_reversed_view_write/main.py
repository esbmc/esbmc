import numpy as np

a = np.array([1, 2, 3])
r = a[::-1]
r[0] = 9
assert a[2] == 9
