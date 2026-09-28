import numpy as np

u = np.unique(np.array([3, 1, 3, 2, 1]))

assert len(u) == 3
assert u[0] == 1
assert u[1] == 2
assert u[2] == 3
