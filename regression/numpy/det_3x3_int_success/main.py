import numpy as np
m = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
x = np.linalg.det(m)
assert x == 1
