import numpy as np

a = np.array([[3, 1], [4, 2]])
r1 = np.sort(a, axis=-1)  # Same as axis=1
r2 = np.sort(a, axis=-2)  # Same as axis=0

assert r1[0, 0] == 1
assert r2[0, 0] == 3
