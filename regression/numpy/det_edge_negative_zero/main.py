import numpy as np
m = np.array([[0.0, 1.0], [0.0, 2.0]])
x = np.linalg.det(m)
assert x == 0.0
