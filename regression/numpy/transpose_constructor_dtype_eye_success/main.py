import numpy as np

base = np.full((2, 2), 4.9, dtype=int)
a = base.transpose()

assert a[0][0] == 4
assert a[0][1] == 4
assert a[1][0] == 4
assert a[1][1] == 4
