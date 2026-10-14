import numpy as np

base = np.array([1.5])
f = np.full_like(base, 1.7, dtype=int)

assert f[0] == 1
