import numpy as np

base = np.full((2, 2), 7, dtype=int)
a = base.transpose()

assert a[0][0] == 7
assert a[0][1] == 7
assert a[1][0] == 7
assert a[1][1] == 7
