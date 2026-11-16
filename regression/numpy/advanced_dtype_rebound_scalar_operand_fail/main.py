import numpy as np

wide = np.array([100], dtype=np.int64)
x = np.array([100], dtype=np.int8)
n = wide[0]
n = 100
assert (x + n)[0] == 200
