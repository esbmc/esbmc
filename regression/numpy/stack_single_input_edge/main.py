import numpy as np

a = np.array([1, 2, 3])
c = np.stack([a])

assert c[0][0] == 1
assert c[0][2] == 3
