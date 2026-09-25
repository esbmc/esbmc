import numpy as np

base = np.zeros((2, 3), dtype=int)
a = base.T

assert a.shape[0] == 3
assert a.shape[1] == 2
assert a[0][0] == 0
