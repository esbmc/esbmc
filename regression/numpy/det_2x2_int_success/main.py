import numpy as np
m = np.array([[1, 2], [3, 4]])
x = np.linalg.det(m)
assert x == -2
