import numpy as np

f = np.full(4, 7)
assert f.shape == (4,)
assert f[3] == 7
g = np.full((2, 3), 1.5)
assert g.shape == (2, 3)
assert g[1][2] == 1.5
