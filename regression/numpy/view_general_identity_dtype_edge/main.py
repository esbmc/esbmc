import numpy as np

i = np.identity(2, dtype=int)
assert i[0][0] == 1
assert i[0][1] == 0
f = np.full(3, 2, dtype=float)
assert f[0] == 2.0
