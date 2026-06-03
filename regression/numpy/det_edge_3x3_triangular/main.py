import numpy as np
m = np.array([[2, 1, 3], [0, -1, 4], [0, 0, 5]])
x = np.linalg.det(m)
assert x == -10
