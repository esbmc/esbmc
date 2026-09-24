import numpy as np

base = np.zeros((2, 2), dtype=int)
a = base.flatten()

assert a[0] == 0
assert a[1] == 0
assert a[2] == 0
assert a[3] == 0
