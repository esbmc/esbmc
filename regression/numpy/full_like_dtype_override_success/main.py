import numpy as np

base = np.array([1, 2, 3])
f = np.full_like(base, 7, dtype=int)

assert f[0] == 7
assert f[1] == 7
assert f[2] == 7
