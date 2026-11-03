import numpy as np

a = np.full(2, 0, dtype=bool)
b = np.full(2, True)
assert not a[1]
assert b[0] and b[1]
