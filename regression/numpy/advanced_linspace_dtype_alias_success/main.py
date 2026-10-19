import numpy as np

a = np.linspace(0, 1, 3, dtype="float64")
b = np.linspace(0, 1, 3, dtype=float)
assert a[1] == 0.5 and b[1] == a[1]
