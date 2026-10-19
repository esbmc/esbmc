import numpy as np

a = np.arange(1, 4, dtype="float64")
b = np.arange(1, 4, dtype=float)
assert a[2] == 3.0 and b[2] == a[2]
