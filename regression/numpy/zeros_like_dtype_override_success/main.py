import numpy as np

base = np.array([1, 2, 3])
z = np.zeros_like(base, dtype=float)

assert z.shape[0] == 3
assert z[0] == 0.0
