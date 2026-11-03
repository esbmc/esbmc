import numpy as np
r = np.arange(5)
w = r[2:4]
w[0] = 9
assert r[2] == 2
