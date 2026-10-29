import numpy as np

b = np.linalg.inv(np.array([[1e200, 0.0], [0.0, 3e-200]]))
assert b[1][1] > 0.0, "radius"
