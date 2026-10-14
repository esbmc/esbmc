import numpy as np
m = np.array([[2, 1], [1, 2]])
x = np.linalg.det(m)
assert x == 3
