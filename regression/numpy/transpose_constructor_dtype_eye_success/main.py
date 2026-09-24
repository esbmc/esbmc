import numpy as np

base = np.eye(3, dtype=int)
a = base.transpose()

assert a[0][0] == 1
assert a[1][1] == 1
assert a[2][2] == 1
assert a[0][1] == 0
