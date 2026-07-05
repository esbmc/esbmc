import numpy as np

a = np.array([1, 2, 3])
b = a.astype(np.float64)

assert b[0] == a[0] + 0.5
