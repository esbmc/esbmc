import numpy as np

f = np.full(3, fill_value=1)
assert len(f) == 3
assert f[2] == 1
g = np.full(shape=(2, 2), fill_value=2.5)
assert g[1][1] == 2.5
w = f[1:]
w[0] = 5
assert f[1] == 5
