import numpy as np

a = np.array([3.5, -1.0, 2.25, -1.0])
a.sort()

assert a[0] == -1.0
assert a[1] == -1.0
assert a[2] == 2.25
assert a[3] == 3.5
