import numpy as np

a = np.ones((2, 3))
assert a[0][0] == 1.0
assert a[1][2] == 1.0

p = np.ones((1, 2))
assert p[0][0] == 1.0
assert p[0][1] == 1.0
