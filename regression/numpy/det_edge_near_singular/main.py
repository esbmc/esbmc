import numpy as np
m = np.array([[1.0, 2.0], [2.0, 4.000000001]])
x = np.linalg.det(m)
assert x < 1e-8
assert x > 0.0
