import numpy as np

q = np.floor_divide(np.array([7, -7]), np.array([2, 2]))
assert q[1] == -3
