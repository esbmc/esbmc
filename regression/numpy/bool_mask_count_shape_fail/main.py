import numpy as np

n = nondet_bool()
a = np.array([1, 5, 2, 8])
mask = np.array([n, True, False, True])
sel = a[mask]
assert len(sel) >= 2
assert len(sel) <= 2
assert sel[len(sel) - 1] == 8
