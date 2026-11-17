import numpy as np

r = np.mod(np.array([-7.5]), 2.0)
assert r[0] == -1.5
