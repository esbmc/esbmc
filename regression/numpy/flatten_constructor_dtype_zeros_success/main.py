import numpy as np

base = np.full((2, 2), 2.9, dtype=int)
a = base.flatten()

assert a[0] == 2
assert a[1] == 2
assert a[2] == 2
assert a[3] == 2
