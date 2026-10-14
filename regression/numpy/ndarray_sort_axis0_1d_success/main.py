import numpy as np

a = np.array([3, 1, 2])
a.sort(axis=0)

assert a[0] == 1
assert a[1] == 2
assert a[2] == 3
