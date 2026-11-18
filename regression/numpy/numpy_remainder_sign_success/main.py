import numpy as np

r = np.remainder(np.array([7, -7, 7, -7]), np.array([2, 2, -2, -2]))
assert r[0] == 1 and r[1] == 1 and r[2] == -1 and r[3] == -1
m = np.mod(np.array([7.5, -7.5]), -2.0)
assert m[0] == -0.5 and m[1] == -1.5
assert np.mod(-7, 3) == np.remainder(-7, 3)
