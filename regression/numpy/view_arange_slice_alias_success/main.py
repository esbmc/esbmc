import numpy as np
r = np.arange(5)
w = r[2:4]
w[0] = 9
assert r[2] == 9
s = r.sum()
assert s == 0 + 1 + 9 + 3 + 4
assert len(r) == 5
