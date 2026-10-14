import numpy as np

a = np.array([3, 1, 4, 1, 5])
a.sort()

assert a[0] == 1
assert a[1] == 1
assert a[2] == 3
assert a[3] == 4
assert a[4] == 5
