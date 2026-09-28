import numpy as np

a = np.empty([1, 2], dtype=bool)
a[0][0] = True
a[0][1] = False

assert a[0][0]
assert not a[0][1]
