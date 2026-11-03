import numpy as np

e = np.eye(2, 4, k=2)
assert e.shape == (2, 4)
assert e[0][2] == 1
assert e[1][3] == 1
assert e[0][0] == 0
