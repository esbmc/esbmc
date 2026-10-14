import numpy as np
m = np.array([[0, 1], [1, 0]])
x = np.linalg.det(m)
assert x == -1
