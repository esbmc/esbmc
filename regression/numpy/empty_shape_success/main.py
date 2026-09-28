import numpy as np

a = np.empty((2, 2), dtype=int)
a[0][0] = 1
a[1][1] = 4

assert a[0][0] == 1
assert a[1][1] == 4
