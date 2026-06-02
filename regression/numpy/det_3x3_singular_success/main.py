import numpy as np
m = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
x = np.linalg.det(m)
assert x == 0
